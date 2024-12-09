import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from paddleocr import PaddleOCR

class SudokuSolver:
    def __init__(self):
        self.grid = None
        self.original_image = None
        # Initialize PaddleOCR with CPU
        self.ocr = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=False)

    def read_image(self, image_path):
        # Read image and preprocess
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
        
        # 查找轮廓
        contours, hierarchy = cv2.findContours(
            dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # 绘制所有找到的轮廓
        contour_img = self.original_image.copy()
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
        cv2.imwrite('debug_5_contours.png', contour_img)
        
        print(f"Found {len(contours)} contours")
        
        return dilated

    def find_grid(self, image):
        # 查找最大的矩形轮廓（应该是数独网格）
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
        
        # Reshape the grid_contour to ensure it's in the correct format
        grid_contour_reshaped = grid_contour.reshape(-1, 2)
        box_reshaped = box.reshape(-1, 2)
        
        cv2.drawContours(grid_img, [grid_contour_reshaped], -1, (0, 255, 0), 2)
        cv2.drawContours(grid_img, [box_reshaped], -1, (0, 0, 255), 2)
        cv2.imwrite('debug_6_grid.png', grid_img)
        
        return box

    def get_grid_corners(self, grid_contour):
        # Compute the convex hull to get the grid corners
        perimeter = cv2.arcLength(grid_contour, True)
        corners = cv2.approxPolyDP(grid_contour, 0.02 * perimeter, True)
        
        # Ensure we have exactly 4 corners
        if len(corners) != 4:
            raise ValueError(f"Expected 4 corners, but found {len(corners)}")
        
        # Flatten the corners array and convert to float32
        corners = corners.reshape(4, 2).astype(np.float32)
        
        # Order points: top-left, top-right, bottom-right, bottom-left
        s = corners.sum(axis=1)
        corners_ordered = np.zeros((4, 2), dtype=np.float32)
        corners_ordered[0] = corners[np.argmin(s)]  # Top-left
        corners_ordered[3] = corners[np.argmax(s)]  # Bottom-right
        diff = np.diff(corners, axis=1)
        corners_ordered[1] = corners[np.argmin(diff)]  # Top-right
        corners_ordered[2] = corners[np.argmax(diff)]  # Bottom-left
        
        # Ensure corners are explicitly converted to float32
        corners_ordered = corners_ordered.astype(np.float32)
        
        return corners_ordered

    def transform_perspective(self, image, corners):
        # Ensure corners are float32
        corners = corners.astype(np.float32)
        
        # Apply perspective transform
        width = height = 450  # Fixed size for transformed image
        dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
        
        matrix = cv2.getPerspectiveTransform(corners, dst)
        warped = cv2.warpPerspective(image, matrix, (width, height))
        
        # 保存透视变换后的图像用于调试
        cv2.imwrite('debug_warped.png', warped)
        print(f"Warped image shape: {warped.shape}")
        
        return warped

    def extract_digits(self, image):
        # 确保输入图像是灰度图
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
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=11,
            C=2
        )
        processed_images.append(thresh1)
        
        # 第二遍：Otsu's阈值
        _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        processed_images.append(thresh2)
        
        # 第三遍：均值阈值+形态学操作
        thresh3 = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=15,
            C=5
        )
        kernel = np.ones((2,2), np.uint8)
        thresh3 = cv2.morphologyEx(thresh3, cv2.MORPH_OPEN, kernel)
        processed_images.append(thresh3)

        # 创建彩色调试图像
        debug_images = [cv2.cvtColor(thresh.copy(), cv2.COLOR_GRAY2BGR) for thresh in processed_images]
        final_debug_image = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2BGR)  # 用于整合所有结果
        
        # 初始化网格
        grid = [[0 for _ in range(9)] for _ in range(9)]
        
        # 存储每个单元格的所有检测结果
        cell_detections = {}

        for i in range(9):
            for j in range(9):
                x = j * cell_width
                y = i * cell_height
                
                # 精确计算单元格坐标
                y1 = y
                y2 = y + cell_height
                x1 = x
                x2 = x + cell_width
                
                detections = []
                colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # 绿色、红色、蓝色
                
                # 对每种处理方式进行OCR
                for pass_num, processed_image in enumerate(processed_images):
                    # Extract cell
                    cell = processed_image[y1:y2, x1:x2]
                    
                    # Save original cell
                    cv2.imwrite(os.path.join(debug_dirs['original'], f'original_cell_{i}_{j}_pass{pass_num+1}.png'), cell)
                    
                    # 寻找数字的实际边界
                    non_zero = cv2.findNonZero(cell)
                    
                    if non_zero is not None and len(non_zero) > 0:
                        # 获取最小外接矩形
                        x_min, y_min, w, h = cv2.boundingRect(non_zero)
                        
                        # 更激进的裁剪策略
                        padding = max(int(min(w, h) * 0.1), 1)
                        x_min = max(0, x_min + padding)
                        y_min = max(0, y_min + padding)
                        w = max(0, w - 2*padding)
                        h = max(0, h - 2*padding)
                        
                        # 裁剪到数字区域
                        cell = cell[y_min:y_min+h, x_min:x_min+w]
                    
                    # 转换为RGB进行OCR
                    cell_rgb = cv2.cvtColor(cell, cv2.COLOR_GRAY2RGB)
                    
                    # OCR处理
                    try:
                        result = self.ocr.ocr(cell_rgb, cls=False)
                        if result and result[0]:
                            for line in result[0]:
                                text = line[1][0]
                                confidence = line[1][1]
                                if confidence > 0.3 and text.isdigit() and 1 <= int(text) <= 9:
                                    digit = int(text)
                                    detections.append((digit, confidence))
                                    
                                    # 在对应的debug图像上标注结果
                                    cv2.putText(debug_images[pass_num], f"{digit}", 
                                              (x1 + cell_width//4, y1 + cell_height//2),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[pass_num], 2)
                    except Exception as e:
                        print(f"Error in OCR pass {pass_num + 1} for cell ({i},{j}): {e}")
                
                # 存储该单元格的所有检测结果
                cell_detections[(i,j)] = detections
                
                # 使用投票机制确定最终数字
                if detections:
                    # Count occurrences of each digit
                    digit_counts = {}
                    digit_confidences = {}
                    for digit, conf in detections:
                        digit_counts[digit] = digit_counts.get(digit, 0) + 1
                        digit_confidences[digit] = max(conf, digit_confidences.get(digit, 0))
                    
                    # 找出检测到的所有数字
                    detected_digits = list(digit_counts.keys())
                    
                    # 如果三次检测到同一个数字，或者至少有两次检测到相同的数字
                    if len(detected_digits) == 1 or max(digit_counts.values()) >= 2:
                        # 如果只有一个数字被检测到，选择这个数字
                        if len(detected_digits) == 1:
                            best_digit = detected_digits[0]
                        else:
                            # 如果多个数字，选择出现最多的数字
                            max_count = max(digit_counts.values())
                            best_digits = [d for d, c in digit_counts.items() if c == max_count]
                            best_digit = max(best_digits, key=lambda d: digit_confidences[d])
                        
                        grid[i][j] = best_digit
                        print(f"Cell ({i},{j}): Recognized {best_digit} with confidence {digit_confidences[best_digit]:.2f}")
                        
                        # 在最终的debug图像上标注结果（绿色表示可信的结果）
                        cv2.putText(final_debug_image, f"{best_digit}", 
                                  (x1 + cell_width//4, y1 + cell_height//2),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        # 如果没有可信的数字，在debug图像上用红色标注（表示不可信）
                        if detections:
                            digit = detections[0][0]
                            cv2.putText(final_debug_image, f"{digit}", 
                                      (x1 + cell_width//4, y1 + cell_height//2),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Save the debug recognition images
        for pass_num, debug_image in enumerate(debug_images):
            cv2.imwrite(f'debug_recognition_pass_{pass_num+1}.png', debug_image)
        
        # 保存最终的debug图像，包含所有结果
        cv2.imwrite('debug_recognition.png', final_debug_image)
        
        return grid

    def is_valid(self, grid, row, col, num):
        # Check if a number is valid in given position
        # Check row
        for x in range(9):
            if grid[row][x] == num:
                return False
        # Check column
        for x in range(9):
            if grid[x][col] == num:
                return False
        # Check 3x3 box
        start_row = row - row % 3
        start_col = col - col % 3
        for i in range(3):
            for j in range(3):
                if grid[i + start_row][j + start_col] == num:
                    return False
        return True

    def solve_sudoku(self, grid):
        # Solve Sudoku using backtracking
        find = self.find_empty(grid)
        if not find:
            return True
        row, col = find

        for num in range(1, 10):
            if self.is_valid(grid, row, col, num):
                grid[row][col] = num
                if self.solve_sudoku(grid):
                    return True
                grid[row][col] = 0
        return False

    def find_empty(self, grid):
        # Find an empty cell in the grid
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 0:
                    return (i, j)
        return None

    def create_solution_image(self, original_grid, solved_grid):
        # Create a new image with the solution
        img = Image.new('RGB', (450, 450), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw grid lines
        for i in range(10):
            line_width = 3 if i % 3 == 0 else 1
            draw.line([(i * 50, 0), (i * 50, 450)], fill='black', width=line_width)
            draw.line([(0, i * 50), (450, i * 50)], fill='black', width=line_width)

        # Add numbers
        try:
            font = ImageFont.truetype("arial.ttf", 32)
        except:
            font = ImageFont.load_default()

        for i in range(9):
            for j in range(9):
                x = j * 50 + 15
                y = i * 50 + 5
                if original_grid[i][j] != 0:
                    # Original numbers in black
                    draw.text((x, y), str(original_grid[i][j]), fill='black', font=font)
                else:
                    # Solved numbers in blue
                    draw.text((x, y), str(solved_grid[i][j]), fill='blue', font=font)
        
        return img

    def process_image(self, image_path, output_path):
        # Main processing pipeline
        print("Reading and preprocessing image...")
        gray = self.read_image(image_path)
        
        print("Finding grid...")
        grid_contour = self.find_grid(gray)
        
        print("Getting grid corners...")
        corners = grid_contour
        
        print("Transforming perspective...")
        warped = self.transform_perspective(gray, corners)
        
        print("Extracting digits...")
        grid = self.extract_digits(warped)
        
        # Print the recognized grid for debugging
        print("\nRecognized Sudoku grid:")
        for row in grid:
            print(row)
        
        # Create a copy for solving
        solved_grid = grid.copy()
        
        print("\nSolving Sudoku...")
        # Solve the Sudoku
        if self.solve_sudoku(solved_grid):
            print("Sudoku solved successfully!")
            # Create and save solution image
            solution_img = self.create_solution_image(grid, solved_grid)
            solution_img.save(output_path)
            print(f"Solution saved as '{output_path}'")
            return True
        else:
            print("Could not solve the Sudoku puzzle!")
            return False

# Example usage
if __name__ == "__main__":
    solver = SudokuSolver()
    input_image = "sudoku.jpg"  
    output_image = "sudoku_solution.png"
    
    if os.path.exists(input_image):
        try:
            if solver.process_image(input_image, output_image):
                print(f"Solution saved to {output_image}")
            else:
                print("Could not solve the Sudoku puzzle")
        except Exception as e:
            print(f"Error processing image: {str(e)}")
    else:
        print(f"Input image {input_image} not found")
