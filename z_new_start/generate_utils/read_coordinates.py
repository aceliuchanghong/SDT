from PIL import Image, ImageDraw


def draw_character_strokes(coordinates, image_size=(256, 256), scale_factor=1):
    normalized_coordinates = normalize_coordinates(coordinates, image_size, scale_factor)
    images = {}
    for char, strokes in normalized_coordinates.items():
        if strokes is None or isinstance(strokes, str):
            continue

        image = Image.new('1', image_size, 1)  # 创建黑白图像
        draw = ImageDraw.Draw(image)
        for stroke in strokes:
            for i in range(len(stroke) - 1):
                x1, y1, _, _ = stroke[i]
                x2, y2, _, _ = stroke[i + 1]
                draw.line(
                    (x1, y1, x2, y2),
                    fill=0, width=2  # 使用黑色线条
                )
        images[char] = image
    return images


def normalize_coordinates(coordinates, image_size, scale_factor):
    normalized = {}
    for char, strokes in coordinates.items():
        if strokes is None or isinstance(strokes, str):
            continue

        all_points = [point for stroke in strokes for point in stroke]
        min_x = min(point[0] for point in all_points)
        min_y = min(point[1] for point in all_points)
        max_x = max(point[0] for point in all_points)
        max_y = max(point[1] for point in all_points)

        width = max_x - min_x
        height = max_y - min_y

        offset_x = (image_size[0] - width * scale_factor) / 2
        offset_y = (image_size[1] - height * scale_factor) / 2

        norm_strokes = []
        for stroke in strokes:
            norm_stroke = [
                ((x - min_x) * scale_factor + offset_x, (image_size[1] - (y - min_y) * scale_factor - offset_y), p1, p2)
                for x, y, p1, p2 in stroke]
            norm_strokes.append(norm_stroke)
        normalized[char] = norm_strokes

    return normalized


if __name__ == '__main__':
    coor = {'一': [
        [(64.0, 385.0, 1, 0), (68.0, 384.0, 0, 0), (78.5, 381.0, 0, 0), (89.0, 378.0, 0, 0), (99.5, 378.0, 0, 0),
         (110.0, 378.0, 0, 0), (837.0, 415.0, 0, 0), (862.0, 417.0, 0, 0), (872.0, 421.0, 0, 0), (882.0, 425.0, 0, 0),
         (887.0, 425.0, 0, 0), (892.0, 425.0, 0, 0), (905.0, 417.0, 0, 0), (918.0, 409.0, 0, 0), (928.5, 397.5, 0, 0),
         (939.0, 386.0, 0, 0), (939.0, 378.0, 0, 0), (939.0, 367.0, 0, 0), (921.0, 365.0, 0, 0), (148.0, 325.0, 0, 0),
         (136.0, 324.0, 0, 0), (133.0, 324.0, 0, 0), (121.0, 324.0, 0, 0), (103.0, 324.0, 0, 0), (94.0, 330.0, 0, 0),
         (78.0, 343.0, 0, 0), (71.0, 361.5, 0, 0), (64.0, 380.0, 0, 1)]], }
    images = draw_character_strokes(coor, scale_factor=0.27)
    for char, image in images.items():
        image.save(f"{char}.png")  # 保存图像
