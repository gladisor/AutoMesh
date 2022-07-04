

    # x = data[7]
    # print(torch.sigmoid(model(x.pos, x.edge_index)).topk(100, dim = 0))
    
    # x = data[6]
    # y_hat = torch.sigmoid(model(x.pos, x.edge_index))

    # print(loss_func(y_hat, x.y))

    # data.display(50)

    # x = data[1]

    # print(x.y.max(dim = 0))

    # r = x.y.topk(5, dim = 0)
    # D = torch.tensor(squareform(pdist(x.pos[r.indices[:, 0]])), dtype = torch.float32)

    # J = torch.ones(D.shape) / D.numel()
    # C = torch.eye(D.shape[0]) - J
    # B = -0.5 * C @ D @ C
    # eigen = torch.linalg.eig(B)
    # m = eigen.eigenvalues.real.topk(2).indices

    # A_m = torch.diag(eigen.eigenvalues.real[m]).sqrt()
    # E_m = eigen.eigenvectors.real[:, m]

    # X = E_m @ A_m

    # print(X)
    # import matplotlib.pyplot as plt

    # plt.scatter(X[:, 0], X[:, 1])
    # plt.show()