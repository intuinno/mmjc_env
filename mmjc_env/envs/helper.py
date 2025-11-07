def render_maze_with_agent_and_targets(
    maze_layout,
    agent_pos,
    agent_dir,
    targets_pos,
    targets_color_rgb=None,  # Expects (N, 3) for N targets
    cell_pixels=30,  # Pixels per maze cell
    wall_color=(100, 50, 20),  # Brown
    floor_color=(128, 128, 128),  # Grey
    agent_color=(255, 255, 255),  # White
    target_radius_ratio=0.35,  # Ratio of cell_pixels for target radius
):
    """
    Renders a NumPy image of the maze, agent, and targets.

    Args:
        maze_layout (np.ndarray): 2D binary array (height, width) where 1=wall, 0=floor.
        agent_pos (np.ndarray): (2,) float array, agent's (x, y) global coordinates.
        agent_dir (np.ndarray): (2,) float array, agent's (dx, dy) unit vector orientation.
        targets_pos (np.ndarray): (N, 2) float array, (x, y) global coordinates of N targets.
        targets_color_rgb (np.ndarray, optional): (N, 3) float array of RGB colors (0-1 range).
                                                 If None, default to common colors.
        cell_pixels (int): Number of pixels per side of a maze cell.
        wall_color (tuple): RGB tuple (0-255) for walls.
        floor_color (tuple): RGB tuple (0-255) for floor.
        agent_color (tuple): RGB tuple (0-255) for the agent.
        target_radius_ratio (float): Radius of target circles as a ratio of cell_pixels.

    Returns:
        np.ndarray: A (height*cell_pixels, width*cell_pixels, 3) NumPy array
                    representing the rendered RGB image.
    """
    maze_height, maze_width = maze_layout.shape
    img_height = maze_height * cell_pixels
    img_width = maze_width * cell_pixels

    # Initialize a blank image with the floor color
    image = np.full((img_height, img_width, 3), floor_color, dtype=np.uint8)

    # Draw the maze walls
    for r in range(maze_height):
        for c in range(maze_width):
            if maze_layout[r, c] == 1:
                y_start, x_start = r * cell_pixels, c * cell_pixels
                image[
                    y_start : y_start + cell_pixels, x_start : x_start + cell_pixels
                ] = wall_color

    # Helper to convert global (x,y) to pixel (px, py) (y-down convention)
    def global_to_pixel(global_x, global_y):
        px = int(global_x * cell_pixels)
        py = int(global_y * cell_pixels)
        return px, py

    # Draw the targets (circles)
    if targets_color_rgb is None:
        # Default colors if not provided
        default_colors = [
            (0, 0, 255),
            (0, 255, 0),
            (255, 0, 0),
            (255, 255, 0),
            (0, 255, 255),
            (255, 0, 255),
        ]  # Blue, Green, Red, Yellow, Cyan, Magenta
        targets_color_rgb = np.array(default_colors[: len(targets_pos)])

    for i, target_pos in enumerate(targets_pos):
        target_x, target_y = target_pos
        center_px, center_py = global_to_pixel(target_x, target_y)
        radius_px = int(cell_pixels * target_radius_ratio)
        target_color = (
            (targets_color_rgb[i] * 255).astype(np.uint8)
            if targets_color_rgb.max() <= 1
            else targets_color_rgb[i].astype(np.uint8)
        )

        # Simple circle drawing by iterating pixels in bounding box
        for y in range(
            max(0, center_py - radius_px), min(img_height, center_py + radius_px + 1)
        ):
            for x in range(
                max(0, center_px - radius_px), min(img_width, center_px + radius_px + 1)
            ):
                if (x - center_px) ** 2 + (y - center_py) ** 2 <= radius_px**2:
                    image[y, x] = target_color

    # Draw the agent (triangle)
    agent_x, agent_y = agent_pos
    center_px, center_py = global_to_pixel(agent_x, agent_y)

    # Calculate triangle points based on agent_dir
    # Head of the triangle
    head_offset_x = agent_dir[0] * cell_pixels * 0.4
    head_offset_y = agent_dir[1] * cell_pixels * 0.4
    p1_x, p1_y = center_px + head_offset_x, center_py + head_offset_y

    # Perpendicular direction for the base of the triangle
    perp_dir_x = -agent_dir[1]  # Rotate 90 deg clockwise
    perp_dir_y = agent_dir[0]

    # Base points
    base_offset_x = perp_dir_x * cell_pixels * 0.25
    base_offset_y = perp_dir_y * cell_pixels * 0.25

    # Move base back slightly from center
    back_offset_x = -agent_dir[0] * cell_pixels * 0.2
    back_offset_y = -agent_dir[1] * cell_pixels * 0.2

    p2_x = center_px - base_offset_x + back_offset_x
    p2_y = center_py - base_offset_y + back_offset_y
    p3_x = center_px + base_offset_x + back_offset_x
    p3_y = center_py + base_offset_y + back_offset_y

    triangle_points = np.array(
        [[p1_x, p1_y], [p2_x, p2_y], [p3_x, p3_y]], dtype=np.int32
    )

    # A very simple way to "draw" a filled triangle (can be improved)
    # This is a brute-force fill, not optimized.
    min_x, max_x = np.min(triangle_points[:, 0]), np.max(triangle_points[:, 0])
    min_y, max_y = np.min(triangle_points[:, 1]), np.max(triangle_points[:, 1])

    for y in range(max(0, min_y - 1), min(img_height, max_y + 2)):
        for x in range(max(0, min_x - 1), min(img_width, max_x + 2)):
            # Check if point (x,y) is inside the triangle
            # This is a basic point-in-triangle test (barycentric coordinates or winding number)
            # For simplicity, we'll draw lines and then fill, but a more robust drawing library
            # would be better here. For now, a rough fill:
            if is_point_in_triangle((x, y), triangle_points):
                image[y, x] = agent_color

    # Draw edges for clarity (simple line drawing, can be jagged)
    def draw_line(img, p1, p2, color, thickness=1):
        x0, y0 = int(p1[0]), int(p1[1])
        x1, y1 = int(p2[0]), int(p2[1])

        # Bresenham's line algorithm or similar for better lines
        # This is a very basic interpolation.
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            if 0 <= y0 < img_height and 0 <= x0 < img_width:
                img[y0, x0] = color
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    draw_line(image, triangle_points[0], triangle_points[1], agent_color, 2)
    draw_line(image, triangle_points[1], triangle_points[2], agent_color, 2)
    draw_line(image, triangle_points[2], triangle_points[0], agent_color, 2)

    return image


# --- Helper function for point-in-triangle test (rough implementation) ---
def is_point_in_triangle(pt, tri_pts):
    # tri_pts: [[x1,y1], [x2,y2], [x3,y3]]
    # pt: (x,y)

    x, y = pt
    x1, y1 = tri_pts[0]
    x2, y2 = tri_pts[1]
    x3, y3 = tri_pts[2]

    # Calculate barycentric coordinates or just signed areas
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    s1 = sign((x, y), (x1, y1), (x2, y2))
    s2 = sign((x, y), (x2, y2), (x3, y3))
    s3 = sign((x, y), (x3, y3), (x1, y1))

    has_neg = (s1 < 0) or (s2 < 0) or (s3 < 0)
    has_pos = (s1 > 0) or (s2 > 0) or (s3 > 0)

    # All signs must be the same (or zero for points on edges)
    return not (has_neg and has_pos)

