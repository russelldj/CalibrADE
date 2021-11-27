import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, distance
import numpy as np
import pickle
import pathlib
import imageio as io

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# np.random.seed(1000)
# N = 20 #num of points
# D = 2 #dimension of points (2D/3D)
# H = 100; W = 100; # img dimensions (keeping same for dummy distribution)

# # initial point distribution (will come from image set control points)
# pts = np.random.uniform(low=0, high=H, size=(N,D))

# # add boundary points (corners and centers of boundaries for now)
# bds = np.array([[0,0],[0,int(H/2)],[0,H],[int(W/2),0],[0,W],[W,H],[W,int(H/2)],[H,int(W/2)]])
# pts = np.concatenate((pts,bds),axis=0)
# # print(pts.shape)

# # calc voronoi centers
# vor = Voronoi(pts)
# idx = []
# for i in range(0,len(vor.vertices)):
#     if(vor.vertices[i][0]>W or vor.vertices[i][0]<0 or vor.vertices[i][1]<0 or vor.vertices[i][1]>H):
#         idx.append(i)

# verts = np.delete(vor.vertices,idx,axis=0)
# print(np.amax(verts))
# # find min distance for each voronoi vertex with its neighbourhood points
# # find the max of the vornoi_vertex-ngb_pt distance - this is the dispersion value
# dist = distance.cdist(pts,verts,metric='euclidean')
# min_dist = np.amin(dist,axis=0)
# min_dist_args = np.array(np.unravel_index(np.argmin(dist,axis=0),dist.shape[0])).T


# print(min_dist_args.shape)
# L2_disp = np.amax(min_dist)
# arg = np.array([min_dist_args[np.argmax(min_dist)],np.argmax(min_dist)])
# print(L2_disp)
# print(pts[arg[0]],verts[arg[1]])

# # fig = voronoi_plot_2d(vor)
# circ = plt.Circle((verts[arg[1]][0],verts[arg[1]][1]),L2_disp,fill=False)
# plt.gca().add_patch(circ)
# plt.plot(pts[:,0],pts[:,1],'b.')
# # plt.plot(verts[:,0],verts[:,1],'k.')
# plt.plot(pts[arg[0]][0][0],pts[arg[0]][0][1],'r*')
# plt.plot(verts[arg[1]][0],verts[arg[1]][1],'r*')

# plt.xlim([0, W])
# plt.ylim([0, H])
# plt.axis('equal')
# plt.show()


def calc_l2_disp(pickle_file="", pts=[], bds=[], do_plot=True):

    if pickle_file != "":  # pts read from pickle file
        with open(pickle_file, "rb") as handle:
            data = pickle.load(handle)
            pts_read = np.array(data["imgpoints"])
            pts = []
            for i in range(0, pts_read.shape[0]):
                for j in range(0, pts_read.shape[1]):
                    pts.append(pts_read[i][j])
            pts = np.array(pts)
            pts = np.squeeze(pts, axis=1)

            N, D = pts.shape

            # img should have been read from pickle file but that doesnt have relative path yet
            img = io.imread(
                "./../data/images/small_board/oneplus5/VID_20211101_151155/image-000340.png"
            )
            H, W, _ = img.shape
    elif pts.size != 0 and bds.size != 0:  # provided pts directly to API
        pts = pts
        N, D = pts.shape
        bds = bds
        if D == 2:
            H = bds[0]
            W = bds[1]
        # add 3d case
    else:  # generate pts randomly for eval
        np.random.seed(1000)
        N = 20  # num of points
        D = 2  # dimension of points (2D/3D)
        H = 100
        W = 100
        # img dimensions (keeping same for dummy distribution)

        pts = np.random.uniform(low=0, high=H, size=(N, D))

    bds_gran = 10
    for i in range(0, bds_gran):  # needs change for 3D
        bdry_pts = np.array(
            [
                [0, i * (H / bds_gran)],
                [W, i * (H / bds_gran)],
                [i * (W / bds_gran), 0],
                [i * (W / bds_gran), H],
            ]
        )
        bdry_pts = np.floor(bdry_pts)
        bdry_pts = bdry_pts.reshape((4, 2))
        pts = np.concatenate((pts, bdry_pts), axis=0)

    # bds = np.array([[0,0],[0,int(H/2)],[0,H],[int(W/2),0],[0,W],[W,H],[W,int(H/2)],[H,int(W/2)]])
    # pts = np.concatenate((pts,bds),axis=0)

    # calc voronoi centers
    vor = Voronoi(pts)
    idx = []
    for i in range(0, len(vor.vertices)):
        if (
            vor.vertices[i][0] > W
            or vor.vertices[i][0] < 0
            or vor.vertices[i][1] < 0
            or vor.vertices[i][1] > H
        ):
            idx.append(i)

    verts = np.delete(vor.vertices, idx, axis=0)

    dist = distance.cdist(pts, verts, metric="euclidean")
    min_dist = np.amin(dist, axis=0)
    min_dist_args = np.array(np.unravel_index(np.argmin(dist, axis=0), dist.shape[0])).T
    arg = np.array([min_dist_args[np.argmax(min_dist)], np.argmax(min_dist)])

    L2_disp = np.amax(min_dist)

    print(L2_disp)

    ######
    if do_plot:  # needs changes for 3D
        circ = plt.Circle((verts[arg[1]][0], verts[arg[1]][1]), L2_disp, fill=False)
        plt.gca().add_patch(circ)
        plt.plot(pts[:, 0], pts[:, 1], "b.")
        plt.plot(pts[arg[0]][0][0], pts[arg[0]][0][1], "r*")
        plt.plot(verts[arg[1]][0], verts[arg[1]][1], "r*")
        plt.xlim([0, W])
        plt.ylim([0, H])
        plt.axis("equal")
        plt.show()

    return L2_disp


if __name__ == "__main__":
    pickle_file = "./../data/images/small_board/oneplus5/VID_20211101_151155/randomrun_25samples_1637562512178997.pickle"
    print(calc_l2_disp(pickle_file))
