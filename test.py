def forward(self, x):
    z = []  # 初始化输出列表，用于存储每层的预测结果
    for i in range(self.nl):  # 遍历每个检测层
        x[i] = self.m[i](x[i])  # 通过卷积层处理输入特征图
        bs, _, ny, nx = x[i].shape  # 获取特征图的形状 (batch size, channels, height, width)
        # 将特征图重塑为 (batch size, num_anchors, num_outputs, grid_height, grid_width) 并调整维度顺序以便于后续处理
        x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        检查是否需要重新生成网格和锚点网格（通常在输入图像尺寸变化时需要）
        xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)  # 将预测分割为 xy（位置）、wh（宽高）和 conf（置信度）
        xy = (xy * 2 + self.grid[i]) * self.stride[i]  # 计算预测的实际位置 (xy)
        wh = (wh * 2) ** 2 * self.anchor_grid[i]  # 计算预测的实际宽高 (wh)
        y = torch.cat((xy, wh, conf), 4)  # 将 xy, wh 和 conf 连接在一起，得到最终的预测输出
        # 将预测输出重塑为 (batch size, num_anchors * grid_height * grid_width, num_outputs)
        z.append(y.view(bs, self.na * nx * ny, self.no))
    return z  # 返回所有检测层的预测结果