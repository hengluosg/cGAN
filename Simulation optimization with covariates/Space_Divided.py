# import numpy as np
# import matplotlib.pyplot as plt
# from itertools import product
# def plot_grid(width, height, l1=1, l2=1):
#     num_width = int(np.ceil(width / l1))  # 沿宽度方向的单元数
#     num_height = int(np.ceil(height / l2))  # 沿高度方向的单元数

#     # 创建一个图形来展示划分
#     fig, ax = plt.subplots()

#     # 绘制矩形网格
#     for i in range(num_width):
#         for j in range(num_height):
#             rect_x = i * l1  # 当前矩形的左上角x坐标
#             rect_y = j * l2  # 当前矩形的左上角y坐标
#             ax.add_patch(plt.Rectangle((rect_x, rect_y), l1, l2, edgecolor="black", facecolor="none"))

#     ax.set_xlim(0, width)
#     ax.set_ylim(0, height)
#     ax.set_aspect('equal', 'box')  # 确保比例保持一致
#     plt.title(f"2D Space Divided into Rectangles ({width}x{height})")
#     plt.show()


# def grid_sample(d, lowbound, upbound, point):
#     n = int(round(point ** (1.0 / d))) +1 # 每个维度上的点数
#     #np.random.seed(42)
#     grid_1d = np.linspace(lowbound + 0.1, upbound - 0.1, n)
#     grid_points = np.array(list(product(grid_1d, repeat=d)))
#     if len(grid_points) > point:
#         indices = np.random.choice(len(grid_points), size=point, replace=False)
#         grid_points = grid_points[indices]
#     return grid_points


# if __name__ == "__main__":
#     d, lowbound, upbound, point = 2 , 0 , 6 , 100
#     grid_points  = grid_sample(d, lowbound, upbound, point)
#     print(grid_points.shape)

#     plot_grid(10, 10 )








# import numpy as np
# import matplotlib.pyplot as plt

# def plot_grid_with_scatter(width, height, l1=2, l2=1, num_points=100):
   
#     num_width = int(np.ceil(width / l1))  # 沿宽度方向的单元数
#     num_height = int(np.ceil(height / l2))  # 沿高度方向的单元数

#     # 生成网格的中心坐标
#     x_centers = np.linspace(l1 / 2, width - l1 / 2, num_width)
#     y_centers = np.linspace(l2 / 2, height - l2 / 2, num_height)

#     # 计算每个格子里的随机点
#     x_points = np.random.uniform(0, width, num_points)
#     y_points = np.random.uniform(0, height, num_points)

#     # 创建一个图形来展示划分
#     fig, ax = plt.subplots()

#     # 绘制矩形网格
#     for i in range(num_width):
#         for j in range(num_height):
#             rect_x = i * l1  # 当前矩形的左上角x坐标
#             rect_y = j * l2  # 当前矩形的左上角y坐标
#             ax.add_patch(plt.Rectangle((rect_x, rect_y), l1, l2, edgecolor="black", facecolor="none"))

#     # 绘制散点
#     ax.scatter(x_points, y_points, c="red", s=10, label="Random Points")

#     # 设置轴范围和标题
#     ax.set_xlim(0, width)
#     ax.set_ylim(0, height)
#     ax.set_aspect('equal', 'box')  # 确保比例保持一致
#     plt.title(" Random Users")
#     plt.show()

# # Example usage
# plot_grid_with_scatter(width=10, height=6, l1=2, l2=1, num_points=200)



import numpy as np
import matplotlib.pyplot as plt

def plot_grid_with_scatter_and_ratio(width, height, l1=1, l2=1, num_points=100):
    
    num_width = int(np.ceil(width / l1))  # 沿宽度方向的单元数
    num_height = int(np.ceil(height / l2))  # 沿高度方向的单元数

    # 生成网格的中心坐标
    x_centers = np.linspace(l1 / 2, width - l1 / 2, num_width)
    y_centers = np.linspace(l2 / 2, height - l2 / 2, num_height)

    # 计算总的格子数
    total_cells = num_width * num_height

    # 每个格子的随机点数
    points_in_cells = np.zeros((num_width, num_height))

    
    users = np.random.uniform(low=0, high=10, size=(num_points, 2))
    x_points = users[:,0]
    y_points = users[:,1]
    

    # x_points = np.random.uniform(0, width, num_points)
    # y_points = np.random.uniform(0, height, num_points)

    for x, y in zip(x_points, y_points):
        
        x_idx = int(x // l1)  
        y_idx = int(y // l2)  
        if x_idx >= num_width: x_idx = num_width - 1
        if y_idx >= num_height: y_idx = num_height - 1
        points_in_cells[x_idx, y_idx] += 1

    
    cell_ratios = points_in_cells / num_points
    
    
    fig, ax = plt.subplots()

    
    for i in range(num_width):
        for j in range(num_height):
            rect_x = i * l1  # 当前矩形的左上角x坐标
            rect_y = j * l2  # 当前矩形的左上角y坐标
            ax.add_patch(plt.Rectangle((rect_x, rect_y), l1, l2, edgecolor="black", facecolor="none"))
            ax.text(rect_x + l1 / 2, rect_y + l2 / 2, f'{cell_ratios[i, j]:.2f}', ha='center', va='center', fontsize=8)
    ax.scatter(x_points, y_points, color="red", s=10, label="Scatter Points")
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal', 'box')  
    plt.title("Scatter Points and Ratios")
    plt.legend()
    plt.show()
    #print(x_points, y_points)
    
    # print("Points in each grid cell and their proportion to the total:")
    # for i in range(num_width):
    #     for j in range(num_height):
    #         print(f"Cell ({i},{j}) - Points: {points_in_cells[i,j]}, Proportion: {cell_ratios[i,j]:.2f}")


    return users, cell_ratios

if __name__ == '__main__':
    users, cell_ratios = plot_grid_with_scatter_and_ratio(width=10, height=10, l1=5, l2=5, num_points=10)
    print(users.shape, cell_ratios)