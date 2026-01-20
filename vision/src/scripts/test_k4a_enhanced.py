#此脚本用于测试Azure Kinect相机的捕捉功能（增强调试版）
import cv2
import numpy as np
from camera import create_camera
import sys

print("Azure Kinect 测试脚本 - 增强调试版")
print("=" * 50)

# 配置参数
SHOW_DEBUG_INFO = True  # 显示调试信息
DEPTH_COLORMAP = cv2.COLORMAP_JET  # 深度图颜色映射
POINT_CLOUD_SCALE = 1000  # 点云可视化缩放

def normalize_depth_for_display(depth_image):
    """
    将深度图像归一化用于显示
    Azure Kinect 深度值范围通常是 0-4000mm
    """
    if depth_image is None or depth_image.size == 0:
        return np.zeros((480, 640), dtype=np.uint8)
    
    # 移除无效深度值（通常为0）
    valid_depth = depth_image[depth_image > 0]
    if len(valid_depth) == 0:
        return np.zeros_like(depth_image, dtype=np.uint8)
    
    # 计算有效深度范围
    min_depth = np.min(valid_depth)
    max_depth = np.max(valid_depth)
    
    print(f"深度范围: {min_depth}mm - {max_depth}mm")
    
    # 将深度值映射到 0-255 范围
    depth_normalized = np.zeros_like(depth_image, dtype=np.uint8)
    mask = depth_image > 0
    depth_normalized[mask] = ((depth_image[mask] - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)
    
    return depth_normalized

def normalize_point_cloud_for_display(point_cloud):
    """
    将3D点云转换为可视化图像
    """
    if point_cloud is None or point_cloud.size == 0:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 取 Z 坐标（深度）进行可视化
    z_values = point_cloud[:, :, 2]
    
    # 移除无效值
    valid_mask = ~np.isnan(z_values) & ~np.isinf(z_values) & (z_values != 0)
    if not np.any(valid_mask):
        return np.zeros((z_values.shape[0], z_values.shape[1], 3), dtype=np.uint8)
    
    valid_z = z_values[valid_mask]
    min_z, max_z = np.min(valid_z), np.max(valid_z)
    
    print(f"点云 Z 范围: {min_z:.2f}m - {max_z:.2f}m")
    
    # 归一化 Z 值
    z_normalized = np.zeros_like(z_values)
    z_normalized[valid_mask] = (z_values[valid_mask] - min_z) / (max_z - min_z)
    
    # 转换为彩色图像
    z_colored = (z_normalized * 255).astype(np.uint8)
    point_cloud_vis = cv2.applyColorMap(z_colored, cv2.COLORMAP_JET)
    
    return point_cloud_vis

def print_capture_info(capture):
    """打印捕获数据的详细信息"""
    print("\n" + "-" * 30)
    print("捕获数据信息:")
    
    # 彩色图像信息
    if hasattr(capture, 'color') and capture.color is not None:
        print(f"原始彩色图像 (color): {capture.color.shape}, dtype: {capture.color.dtype}")
        print(f"  - 分辨率: {capture.color.shape[1]}x{capture.color.shape[0]}")
        print(f"  - 通道数: {capture.color.shape[2]} (BGRA)")
        print(f"  - 数据范围: {np.min(capture.color)} - {np.max(capture.color)}")
    
    # 变换后的彩色图像信息
    if hasattr(capture, 'transformed_color') and capture.transformed_color is not None:
        print(f"变换彩色图像 (transformed_color): {capture.transformed_color.shape}, dtype: {capture.transformed_color.dtype}")
        print(f"  - 说明: 经过深度相机坐标系变换的彩色图像")
        print(f"  - 用途: 与深度图像对齐，用于RGB-D处理")
    
    # 深度图像信息
    if hasattr(capture, 'depth') and capture.depth is not None:
        print(f"深度图像 (depth): {capture.depth.shape}, dtype: {capture.depth.dtype}")
        valid_depth = capture.depth[capture.depth > 0]
        if len(valid_depth) > 0:
            print(f"  - 有效深度范围: {np.min(valid_depth)} - {np.max(valid_depth)} mm")
            print(f"  - 有效像素数: {len(valid_depth)} / {capture.depth.size}")
        else:
            print("  - ⚠️ 没有有效的深度数据")
    
    # 点云信息
    if hasattr(capture, 'transformed_depth_point_cloud') and capture.transformed_depth_point_cloud is not None:
        pc = capture.transformed_depth_point_cloud
        print(f"深度点云 (transformed_depth_point_cloud): {pc.shape}, dtype: {pc.dtype}")
        print(f"  - 说明: 3D点云数据 (X, Y, Z 坐标)")
        
        # 检查点云有效性
        valid_points = ~np.isnan(pc) & ~np.isinf(pc) & (pc != 0)
        valid_count = np.sum(np.all(valid_points, axis=2))
        print(f"  - 有效3D点数: {valid_count} / {pc.shape[0] * pc.shape[1]}")
        
        if valid_count > 0:
            x_vals = pc[:, :, 0][valid_points[:, :, 0]]
            y_vals = pc[:, :, 1][valid_points[:, :, 1]]
            z_vals = pc[:, :, 2][valid_points[:, :, 2]]
            
            print(f"  - X 范围: {np.min(x_vals):.3f} - {np.max(x_vals):.3f} m")
            print(f"  - Y 范围: {np.min(y_vals):.3f} - {np.max(y_vals):.3f} m")
            print(f"  - Z 范围: {np.min(z_vals):.3f} - {np.max(z_vals):.3f} m")

# 创建相机对象
try:
    cam = create_camera("azure_kinect")
    print("✅ Azure Kinect 相机创建成功")
except Exception as e:
    print(f"❌ 创建相机失败: {e}")
    sys.exit(1)

# 启动相机
try:
    cam.start()
    print("✅ 相机启动成功")
    print("\n控制说明:")
    print("- 按 'q' 或 ESC 退出")
    print("- 按 'd' 切换调试信息显示")
    print("- 按 's' 保存当前帧")
except Exception as e:
    print(f"❌ 启动相机失败: {e}")
    sys.exit(1)

frame_count = 0
save_count = 0

# 设置窗口属性 - 只需要设置一次
window_names = {
    'color': 'Color Camera',
    'transformed_color': 'Transformed Color',
    'depth_colored': 'Depth (Colored)',
    'depth_gray': 'Depth (Grayscale)',
    'point_cloud': 'Point Cloud'
}

# 创建和配置所有窗口
for window_name in window_names.values():
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 480)

print("\n开始图像采集...")

try:
    while True:
        try:
            capture = cam.capture()
            frame_count += 1
            
            # 每30帧显示一次调试信息
            if SHOW_DEBUG_INFO and frame_count % 30 == 1:
                print_capture_info(capture)
            
            # 显示原始彩色图像 (color)
            if hasattr(capture, 'color') and capture.color is not None:
                color_bgr = capture.color[:, :, :3]  # 移除 alpha 通道，转为 BGR
                if color_bgr.shape[0] > 0 and color_bgr.shape[1] > 0:  # 检查图像有效性
                    cv2.imshow(window_names['color'], color_bgr)
            
            # 显示变换后的彩色图像 (transformed_color)
            if hasattr(capture, 'transformed_color') and capture.transformed_color is not None:
                transformed_color_bgr = capture.transformed_color[:, :, :3]
                if transformed_color_bgr.shape[0] > 0 and transformed_color_bgr.shape[1] > 0:
                    cv2.imshow(window_names['transformed_color'], transformed_color_bgr)
            
            # 处理和显示深度图像
            if hasattr(capture, 'depth') and capture.depth is not None:
                # 原始深度图像（归一化显示）
                depth_normalized = normalize_depth_for_display(capture.depth)
                if depth_normalized.shape[0] > 0 and depth_normalized.shape[1] > 0:
                    depth_colored = cv2.applyColorMap(depth_normalized, DEPTH_COLORMAP)
                    cv2.imshow(window_names['depth_colored'], depth_colored)
                    cv2.imshow(window_names['depth_gray'], depth_normalized)
            
            # 处理和显示点云
            if hasattr(capture, 'transformed_depth_point_cloud') and capture.transformed_depth_point_cloud is not None:
                point_cloud_vis = normalize_point_cloud_for_display(capture.transformed_depth_point_cloud)
                if point_cloud_vis.shape[0] > 0 and point_cloud_vis.shape[1] > 0:
                    cv2.imshow(window_names['point_cloud'], point_cloud_vis)
            
            # 在一个窗口显示帧计数
            if frame_count % 30 == 0:
                print(f"已处理 {frame_count} 帧")
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC 或 q
                break
            elif key == ord('d'):  # 切换调试信息
                SHOW_DEBUG_INFO = not SHOW_DEBUG_INFO
                print(f"调试信息显示: {'开启' if SHOW_DEBUG_INFO else '关闭'}")
            elif key == ord('s'):  # 保存图像
                save_count += 1
                if hasattr(capture, 'color'):
                    cv2.imwrite(f'saved_color_{save_count}.jpg', capture.color[:, :, :3])
                if hasattr(capture, 'depth'):
                    depth_normalized = normalize_depth_for_display(capture.depth)
                    cv2.imwrite(f'saved_depth_{save_count}.jpg', depth_normalized)
                print(f"已保存第 {save_count} 组图像")
                
        except Exception as e:
            print(f"捕获帧时出错: {e}")
            continue

except KeyboardInterrupt:
    print("\n用户中断程序")
    
finally:
    print("\n清理资源...")
    cam.stop()
    cv2.destroyAllWindows()
    print("程序结束")

print(f"\n总共处理了 {frame_count} 帧")
print(f"保存了 {save_count} 组图像")

print("\n" + "=" * 50)
print("图像类型解释:")
print("=" * 50)
print("1. 原始彩色图像 (color):")
print("   - 来自RGB摄像头，分辨率根据配置而定")
print("   - 格式: BGRA，4通道")
print("   - 坐标系: RGB摄像头坐标系")
print("")
print("2. 变换彩色图像 (transformed_color):")
print("   - 将彩色图像变换到深度摄像头坐标系")
print("   - 与深度图像像素对齐")
print("   - 用于创建RGB-D图像")
print("")
print("3. 深度图像 (depth):")
print("   - 每个像素表示到传感器的距离(mm)")
print("   - 16位无符号整数")
print("   - 0值表示无效/无法测量的距离")
print("")
print("4. 深度点云 (transformed_depth_point_cloud):")
print("   - 3D坐标 (X, Y, Z)，单位米")
print("   - 每个像素对应一个3D点")
print("   - 可用于3D重建和空间分析")
