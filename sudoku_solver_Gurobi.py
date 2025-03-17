import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from paddleocr import PaddleOCR
import logging
from datetime import datetime
import gurobipy as gp
from gurobipy import GRB
from typing import List, Dict, Tuple, Optional, Set

class SudokuSolver:
    def __init__(self, debug_mode=False):
        """初始化数独求解器"""
        self.debug_mode = debug_mode
        self.debug_dir = "sudoku_logs"
        self.grid = None
        self.original_image = None
        
        # 初始化PaddleOCR
        self.ocr = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=False)
        
        # 创建日志目录
        if debug_mode and not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir)
            
        # 设置日志
        self.log_file = os.path.join(self.debug_dir, f"sudoku_solver_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        logging.basicConfig(
            filename=self.log_file,
            level=logging.DEBUG if debug_mode else logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def log(self, message: str):
        """记录日志消息"""
        if self.debug_mode:
            print(message)
            self.logger.debug(message)

    def save_debug_info(self, stage: str, info: str):
        """保存调试信息
        
        Args:
            stage: 处理阶段
            info: 调试信息
        """
        if self.debug_mode:
            debug_file = os.path.join(self.debug_dir, f"debug_{stage}.txt")
            with open(debug_file, 'a', encoding='utf-8') as f:
                f.write(f"{datetime.now()}: {info}\n")

    def process_and_solve(self, image_path: str, output_path: str) -> bool:
        """完整的处理和求解流程
        
        Args:
            image_path: 输入图像路径
            output_path: 输出图像路径
        Returns:
            是否成功处理和求解
        """
        try:
            # 1. 图像处理阶段
            self.save_debug_info("process", "Starting image processing")
            gray = self.read_image(image_path)
            grid_contour = self.find_grid(gray)
            warped = self.transform_perspective(gray, grid_contour)
            grid = self.extract_digits(warped)
            
            # 2. 打印识别结果
            self.save_debug_info("recognition", "Recognized grid:")
            grid_str = "\n".join([" ".join(map(str, row)) for row in grid])
            self.save_debug_info("recognition", grid_str)
            
            # 3. 求解阶段
            solved_grid = [row[:] for row in grid]
            if self.solve_sudoku(solved_grid):
                # 4. 创建并保存结果
                solution_img = self.create_solution_image(grid, solved_grid)
                solution_img.save(output_path)
                
                # 5. 保存最终结果
                self.save_debug_info("solution", "Final solution:")
                solution_str = "\n".join([" ".join(map(str, row)) for row in solved_grid])
                self.save_debug_info("solution", solution_str)
                
                return True
            else:
                self.save_debug_info("error", "Could not solve the puzzle")
                return False
            
        except Exception as e:
            self.save_debug_info("error", f"Error occurred: {str(e)}")
            raise

    def read_image(self, image_path):
        """读取并预处理图像"""
        print(f"Reading image from: {image_path}")
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not read the image from {image_path}")
        
        # 保存原始图像用于调试
        cv2.imwrite('debug_original.png', self.original_image)
        print(f"Original image shape: {self.original_image.shape}")
        
        # 调整图像大小为标准大小
        height, width = self.original_image.shape[:2]
        target_size = 1000
        scale = target_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        self.original_image = cv2.resize(self.original_image, (new_width, new_height))
        
        # 转换为灰度图
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('debug_1_gray.png', gray)
        
        # 高斯模糊减少噪声
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        cv2.imwrite('debug_2_blurred.png', blurred)
        
        # 自适应阈值处理
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        cv2.imwrite('debug_3_thresh.png', thresh)
        
        # 膨胀操作，使数字更清晰
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        cv2.imwrite('debug_4_dilated.png', dilated)
        
        return dilated

    def find_grid(self, image):
        """查找最大的矩形轮廓（数独网格）"""
        contours, hierarchy = cv2.findContours(
            image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            raise ValueError("No contours found in the image")
        
        # 按面积排序找到最大的轮廓
        grid_contour = max(contours, key=cv2.contourArea)
        
        # 确保轮廓面积足够大（至少是图像面积的50%）
        image_area = image.shape[0] * image.shape[1]
        grid_area = cv2.contourArea(grid_contour)
        if grid_area < 0.5 * image_area:
            raise ValueError("Grid contour too small")
        
        # 使用minAreaRect获取最小外接矩形
        rect = cv2.minAreaRect(grid_contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        
        # 绘制找到的网格轮廓和最小外接矩形
        grid_img = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
        grid_contour_reshaped = grid_contour.reshape(-1, 2)
        box_reshaped = box.reshape(-1, 2)
        
        cv2.drawContours(grid_img, [grid_contour_reshaped], -1, (0, 255, 0), 2)
        cv2.drawContours(grid_img, [box_reshaped], -1, (0, 0, 255), 2)
        cv2.imwrite('debug_6_grid.png', grid_img)
        
        return box

    def transform_perspective(self, image, corners):
        """应用透视变换"""
        corners = corners.astype(np.float32)
        width = height = 450
        dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
        
        matrix = cv2.getPerspectiveTransform(corners, dst)
        warped = cv2.warpPerspective(image, matrix, (width, height))
        
        cv2.imwrite('debug_warped.png', warped)
        print(f"Warped image shape: {warped.shape}")
        
        return warped

    def extract_digits(self, image):
        """从图像中提取数字"""
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        height, width = image.shape[:2]
        cell_height = height // 9
        cell_width = width // 9

        # 创建调试目录
        debug_dirs = {
            'original': 'debug_cells_original',
            'processed': 'debug_cells_processed',
            'debug': 'debug_cells'
        }
        for dir_name in debug_dirs.values():
            os.makedirs(dir_name, exist_ok=True)

        # 创建三个不同处理方式的图像
        processed_images = []
        
        # 第一遍：标准自适应阈值
        thresh1 = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        processed_images.append(thresh1)
        
        # 第二遍：Otsu's阈值
        _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        processed_images.append(thresh2)
        
        # 第三遍：均值阈值+形态学操作
        thresh3 = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, 15, 5
        )
        kernel = np.ones((2,2), np.uint8)
        thresh3 = cv2.morphologyEx(thresh3, cv2.MORPH_OPEN, kernel)
        processed_images.append(thresh3)

        # 初始化结果
        grid = [[0 for _ in range(9)] for _ in range(9)]
        debug_images = [cv2.cvtColor(thresh.copy(), cv2.COLOR_GRAY2BGR) for thresh in processed_images]
        final_debug_image = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2BGR)

        for i in range(9):
            for j in range(9):
                x = j * cell_width
                y = i * cell_height
                y1, y2 = y, y + cell_height
                x1, x2 = x, x + cell_width
                
                detections = []
                colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # 绿色、红色、蓝色
                
                # 对每种处理方式进行OCR
                for pass_num, processed_image in enumerate(processed_images):
                    cell = processed_image[y1:y2, x1:x2]
                    cv2.imwrite(os.path.join(debug_dirs['original'], f'original_cell_{i}_{j}_pass{pass_num+1}.png'), cell)
                    
                    # 寻找数字的实际边界
                    non_zero = cv2.findNonZero(cell)
                    if non_zero is not None and len(non_zero) > 0:
                        x_min, y_min, w, h = cv2.boundingRect(non_zero)
                        padding = max(int(min(w, h) * 0.1), 1)
                        x_min = max(0, x_min + padding)
                        y_min = max(0, y_min + padding)
                        w = max(0, w - 2*padding)
                        h = max(0, h - 2*padding)
                        cell = cell[y_min:y_min+h, x_min:x_min+w]
                    
                    # OCR处理
                    try:
                        cell_rgb = cv2.cvtColor(cell, cv2.COLOR_GRAY2RGB)
                        result = self.ocr.ocr(cell_rgb, cls=False)
                        if result and result[0]:
                            for line in result[0]:
                                text = line[1][0]
                                confidence = line[1][1]
                                if confidence > 0.3 and text.isdigit() and 1 <= int(text) <= 9:
                                    digit = int(text)
                                    detections.append((digit, confidence))
                                    cv2.putText(debug_images[pass_num], f"{digit}", 
                                              (x1 + cell_width//4, y1 + cell_height//2),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[pass_num], 2)
                    except Exception as e:
                        print(f"Error in OCR pass {pass_num + 1} for cell ({i},{j}): {e}")
                
                # 使用投票机制确定最终数字
                if detections:
                    digit_counts = {}
                    digit_confidences = {}
                    for digit, conf in detections:
                        digit_counts[digit] = digit_counts.get(digit, 0) + 1
                        digit_confidences[digit] = max(conf, digit_confidences.get(digit, 0))
                    
                    detected_digits = list(digit_counts.keys())
                    if len(detected_digits) == 1 or max(digit_counts.values()) >= 2:
                        if len(detected_digits) == 1:
                            best_digit = detected_digits[0]
                        else:
                            max_count = max(digit_counts.values())
                            best_digits = [d for d, c in digit_counts.items() if c == max_count]
                            best_digit = max(best_digits, key=lambda d: digit_confidences[d])
                        
                        grid[i][j] = best_digit
                        print(f"Cell ({i},{j}): Recognized {best_digit} with confidence {digit_confidences[best_digit]:.2f}")
                        cv2.putText(final_debug_image, f"{best_digit}", 
                                  (x1 + cell_width//4, y1 + cell_height//2),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        if detections:
                            digit = detections[0][0]
                            cv2.putText(final_debug_image, f"{digit}", 
                                      (x1 + cell_width//4, y1 + cell_height//2),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 保存调试图像
        for pass_num, debug_image in enumerate(debug_images):
            cv2.imwrite(f'debug_recognition_pass_{pass_num+1}.png', debug_image)
        cv2.imwrite('debug_recognition.png', final_debug_image)
        
        return grid

    def solve_sudoku(self, grid: List[List[int]]) -> bool:
        """求解数独"""
        print("\nSolving Sudoku...")
        original_grid = [row[:] for row in grid]
        
        # 1. 首先尝试Gurobi
        try:
            print("Attempting solution with Gurobi optimization...")
            solved_grid = self.solve_with_gurobi(grid)
            if solved_grid and self.verify_solution(solved_grid):
                # 如果规则验证通过，就采用这个解
                for i in range(9):
                    for j in range(9):
                        grid[i][j] = solved_grid[i][j]
                print("\nSuccessfully solved using Gurobi!")
                return True
        except Exception as e:
            print(f"Gurobi solver failed: {str(e)}")
        
        # 2. 如果Gurobi失败，尝试回溯
        print("Attempting solution with backtracking...")
        for i in range(9):
            for j in range(9):
                grid[i][j] = original_grid[i][j]
        
        if self.solve_with_backtracking(grid) and self.verify_solution(grid):
            print("\nSuccessfully solved using Backtracking!")
            return True
        
        # 3. 如果两种方法都失败，恢复原始网格
        for i in range(9):
            for j in range(9):
                grid[i][j] = original_grid[i][j]
        
        print("No valid solution found.")
        return False

    def solve_with_gurobi(self, grid: List[List[int]]) -> Optional[List[List[int]]]:
        """使用Gurobi求解数独"""
        try:
            # 创建模型
            model = gp.Model("sudoku")
            model.setParam('OutputFlag', 0)
            
            # 创建变量
            x = {}
            for i in range(9):
                for j in range(9):
                    for k in range(1, 10):
                        x[i,j,k] = model.addVar(vtype=GRB.BINARY, name=f'x_{i}_{j}_{k}')
            
            # 1. 已知数字约束 - 加强约束条件
            for i in range(9):
                for j in range(9):
                    if grid[i][j] != 0:
                        # 确保已知数字位置只能是原有数字
                        model.addConstr(x[i,j,grid[i][j]] == 1)
                        # 其他数字不能出现在这个位置
                        for k in range(1, 10):
                            if k != grid[i][j]:
                                model.addConstr(x[i,j,k] == 0)
            
            # 2. 每个单元格必须且只能填入一个数字
            for i in range(9):
                for j in range(9):
                    model.addConstr(gp.quicksum(x[i,j,k] for k in range(1, 10)) == 1)
            
            # 3. 行约束 - 每个数字在每行只能出现一次
            for i in range(9):
                for k in range(1, 10):
                    model.addConstr(gp.quicksum(x[i,j,k] for j in range(9)) == 1)
            
            # 4. 列约束 - 每个数字在每列只能出现一次
            for j in range(9):
                for k in range(1, 10):
                    model.addConstr(gp.quicksum(x[i,j,k] for i in range(9)) == 1)
            
            # 5. 3x3宫格约束 - 每个数字在每个宫格只能出现一次
            for box_i in range(3):
                for box_j in range(3):
                    for k in range(1, 10):
                        model.addConstr(
                            gp.quicksum(x[i,j,k] 
                                      for i in range(3*box_i, 3*box_i+3)
                                      for j in range(3*box_j, 3*box_j+3)) == 1
                        )
            
            # 求解
            model.optimize()
            
            if model.status == GRB.OPTIMAL:
                # 提取结果并验证
                solution = [[0 for _ in range(9)] for _ in range(9)]
                for i in range(9):
                    for j in range(9):
                        for k in range(1, 10):
                            if x[i,j,k].x > 0.5:
                                solution[i][j] = k
            
                # 验证解的正确性
                if self.verify_solution(solution):
                    print("Gurobi solution verified by rules!")
                    return solution
                else:
                    print("Gurobi solution failed rules verification!")
                    return None
            else:
                print("Gurobi could not find a solution.")
                return None
            
        except gp.GurobiError as e:
            print(f"Gurobi error occurred: {str(e)}")
            return None

    def verify_solution(self, grid: List[List[int]]) -> bool:
        """验证数独解的正确性"""
        try:
            # 1. 检查是否有空格
            for i in range(9):
                for j in range(9):
                    if grid[i][j] == 0:
                        print(f"Empty cell found at position ({i+1},{j+1})")
                        return False
                    if not 1 <= grid[i][j] <= 9:
                        print(f"Invalid number {grid[i][j]} at position ({i+1},{j+1})")
                        return False

            # 2. 检查每行
            for i in range(9):
                row = grid[i]
                if len(set(row)) != 9:
                    print(f"Row {i+1} contains duplicates: {row}")
                    return False

            # 3. 检查每列
            for j in range(9):
                col = [grid[i][j] for i in range(9)]
                if len(set(col)) != 9:
                    print(f"Column {j+1} contains duplicates: {col}")
                    return False

            # 4. 检查每个3x3宫格
            for box_i in range(3):
                for box_j in range(3):
                    box = []
                    for i in range(3*box_i, 3*box_i+3):
                        for j in range(3*box_j, 3*box_j+3):
                            box.append(grid[i][j])
                    if len(set(box)) != 9:
                        print(f"Box at ({box_i+1},{box_j+1}) contains duplicates: {box}")
                        return False

            return True
        except Exception as e:
            print(f"Error in solution verification: {str(e)}")
            return False

    def solve_with_backtracking(self, grid: List[List[int]]) -> bool:
        """使用回溯算法求解数独"""
        find = self.find_empty(grid)
        if not find:
            # 找到解后立即验证
            if self.verify_solution(grid):
                print("Backtracking solution verified by rules!")
                return True
            return False
        
        row, col = find
        for num in range(1, 10):
            if self.is_valid(grid, row, col, num):
                grid[row][col] = num
                if self.solve_with_backtracking(grid):
                    return True
                grid[row][col] = 0
        return False

    def find_empty(self, grid: List[List[int]]) -> Optional[Tuple[int, int]]:
        """查找空单元格"""
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 0:
                    return (i, j)
        return None

    def is_valid(self, grid: List[List[int]], row: int, col: int, num: int) -> bool:
        """检查在给定位置放置数字是否有效"""
        # 检查行
        for x in range(9):
            if grid[row][x] == num:
                return False
                
        # 检查列
        for x in range(9):
            if grid[x][col] == num:
                return False
                
        # 检查3x3宫格
        start_row = row - row % 3
        start_col = col - col % 3
        for i in range(3):
            for j in range(3):
                if grid[i + start_row][j + start_col] == num:
                    return False
        return True

    def create_solution_image(self, original_grid, solved_grid):
        """创建解决方案图像"""
        img = Image.new('RGB', (450, 450), color='white')
        draw = ImageDraw.Draw(img)
        
        # 绘制网格线
        for i in range(10):
            line_width = 3 if i % 3 == 0 else 1
            draw.line([(i * 50, 0), (i * 50, 450)], fill='black', width=line_width)
            draw.line([(0, i * 50), (450, i * 50)], fill='black', width=line_width)

        # 添加数字
        try:
            font = ImageFont.truetype("arial.ttf", 32)
        except:
            font = ImageFont.load_default()

        for i in range(9):
            for j in range(9):
                x = j * 50 + 15
                y = i * 50 + 5
                if original_grid[i][j] != 0:
                    # 原始数字用黑色
                    draw.text((x, y), str(original_grid[i][j]), fill='black', font=font)
                else:
                    # 解出的数字用蓝色
                    draw.text((x, y), str(solved_grid[i][j]), fill='blue', font=font)
        
        return img

if __name__ == "__main__":
    try:
        # 设置输入输出路径
        input_image = "sudoku.jpg"
        output_image = "sudoku_solution.png"
        
        # 检查输入文件是否存在
        if not os.path.exists(input_image):
            print(f"Error: Input image '{input_image}' not found")
            exit(1)
            
        # 创建求解器实例
        solver = SudokuSolver(debug_mode=True)
        
        # 处理和求解
        print("Starting Sudoku processing and solving...")
        if solver.process_and_solve(input_image, output_image):
            print(f"Successfully solved! Solution saved to {output_image}")
            print("Check the logs directory for detailed processing information.")
        else:
            print("Failed to solve the Sudoku puzzle.")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)
