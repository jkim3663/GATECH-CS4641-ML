import numpy as np
from typing import Tuple


class ImgCompression(object):
    def __init__(self):
        pass

    def svd(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  # [4pts]
        """
        Do SVD. You could use numpy SVD.
        Your function should be able to handle black and white
        images ((N,D) arrays) as well as color images ((N,D,3) arrays)
        In the image compression, we assume that each column of the image is a feature. Perform SVD on the channels of
        each image (1 channel for black and white and 3 channels for RGB)
        Image is the matrix X.

        Args:
            X: (N,D) numpy array corresponding to black and white images / (N,D,3) numpy array for color images

        Return:
            U: (N,N) numpy array for black and white images / (N,N,3) numpy array for color images
            S: (min(N,D), ) numpy array for black and white images / (min(N,D),3) numpy array for color images
            V^T: (D,D) numpy array for black and white images / (D,D,3) numpy array for color images
        """
        if len(X.shape) == 2:
            u, s, vh = np.linalg.svd(X, full_matrices=True)
        else:
            ur, sr, vr = np.linalg.svd(X[:,:,0], full_matrices=True)
            ug, sg, vg = np.linalg.svd(X[:,:,1], full_matrices=True)
            ub, sb, vb = np.linalg.svd(X[:,:,2], full_matrices=True)
            
            u = np.stack((ur, ug, ub), axis=-1)
            s = np.stack((sr, sg, sb), axis=-1)
            vh = np.stack((vr, vg, vb), axis=-1)
            
        return u, s, vh

    def compress(
        self, U: np.ndarray, S: np.ndarray, V: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  # [4pts]
        """Compress the SVD factorization by keeping only the first k components

        Args:
            U (np.ndarray): (N,N) numpy array for black and white simages / (N,N,3) numpy array for color images
            S (np.ndarray): (min(N,D), ) numpy array for black and white images / (min(N,D),3) numpy array for color images
            V (np.ndarray): (D,D) numpy array for black and white images / (D,D,3) numpy array for color images
            k (int): int corresponding to number of components to keep

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                U_compressed: (N, k) numpy array for black and white images / (N, k, 3) numpy array for color images
                S_compressed: (k, ) numpy array for black and white images / (k, 3) numpy array for color images
                V_compressed: (k, D) numpy array for black and white images / (k, D, 3) numpy array for color images
        """
        Uc = U[:,0:k]
        Sc = S[:k]
        Vc = V[0:k,:]
        
        return (Uc, Sc, Vc)

    def rebuild_svd(
        self,
        U_compressed: np.ndarray,
        S_compressed: np.ndarray,
        V_compressed: np.ndarray,
    ) -> np.ndarray:  # [4pts]
        """
        Rebuild original matrix X from U, S, and V which have been compressed to k componments.

        Args:
            U_compressed: (N,k) numpy array for black and white images / (N,k,3) numpy array for color images
            S_compressed: (k, ) numpy array for black and white images / (k,3) numpy array for color images
            V_compressed: (k,D) numpy array for black and white images / (k,D,3) numpy array for color images

        Return:
            Xrebuild: (N,D) numpy array of reconstructed image / (N,D,3) numpy array for color images

        Hint: numpy.matmul may be helpful for reconstructing color images
        """
        k = S_compressed.shape[0]
        
        if len(U_compressed.shape) == 2:
            smatrix = np.zeros((k,k))
            for i in range(k):
                smatrix[i, i] = S_compressed[i]
            return np.matmul(np.matmul(U_compressed, smatrix), V_compressed)
        else:
            X_rebuild = np.zeros((U_compressed.shape[0], V_compressed.shape[1], 3))
            smatrix = np.zeros((k, k, 3))
            for i in range(k):
                smatrix[i, i, 0] = S_compressed[i, 0]
                smatrix[i, i, 1] = S_compressed[i, 1]
                smatrix[i, i, 2] = S_compressed[i, 2]
            
            X_rebuild[:,:,0] = np.matmul(np.matmul(U_compressed[:,:,0], smatrix[:,:,0]),V_compressed[:,:,0])
            X_rebuild[:,:,1] = np.matmul(np.matmul(U_compressed[:,:,1], smatrix[:,:,1]),V_compressed[:,:,1])
            X_rebuild[:,:,2] = np.matmul(np.matmul(U_compressed[:,:,2], smatrix[:,:,2]),V_compressed[:,:,2])
            
            return X_rebuild

    def compression_ratio(self, X: np.ndarray, k: int) -> float:  # [4pts]
        """
        Compute the compression ratio of an image: (num stored values in compressed)/(num stored values in original)

        Args:
            X: (N,D) numpy array corresponding to black and white images / (N,D,3) numpy array for color images
            k: int corresponding to number of components

        Return:
            compression_ratio: float of proportion of storage used by compressed image
        """
        total_size = X.shape[0] * X.shape[1]
        comp_size = k * (1 + X.shape[0] + X.shape[1])
        return comp_size / total_size

    def recovered_variance_proportion(self, S: np.ndarray, k: int) -> float:  # [4pts]
        """
        Compute the proportion of the variance in the original matrix recovered by a rank-k approximation

        Args:
           S: (min(N,D), ) numpy array black and white images / (min(N,D),3) numpy array for color images
           k: int, rank of approximation

        Return:
           recovered_var: float (array of 3 floats for color image) corresponding to proportion of recovered variance
        """
        N = S.shape[0]
        # S[i] = sigma_i
        # calculate S[i] for each i in S to get the variance recovered for each S[i]: vector of the same shape as S
        # sum up the first k variances
        # [5, 4, 1, 0.1]
        # k=2, recovered_var = 5+4 = 9
        
        kSum = sum(S[:k] ** 2)
        sigmaSum = sum(S ** 2)
        
        return kSum / sigmaSum
    
    def memory_savings(
        self, X: np.ndarray, U: np.ndarray, S: np.ndarray, V: np.ndarray, k: int
    ) -> Tuple[int, int, int]:
        """
        PROVIDED TO STUDENTS
        
        Returns the memory required to store the original image X and 
        the memory required to store the compressed SVD factorization of X

        Args:
            X (np.ndarray): (N,D) numpy array corresponding to black and white images / (N,D,3) numpy array for color images
            U (np.ndarray): (N,N) numpy array for black and white simages / (N,N,3) numpy array for color images
            S (np.ndarray): (min(N,D), ) numpy array for black and white images / (min(N,D),3) numpy array for color images
            V (np.ndarray): (D,D) numpy array for black and white images / (D,D,3) numpy array for color images
            k (int): integer number of components

        Returns:
            Tuple[int, int, int]: 
                original_nbytes: number of bytes that numpy uses to represent X
                compressed_nbytes: number of bytes that numpy uses to represent U_compressed, S_compressed, and V_compressed
                savings: difference in number of bytes required to represent X 
        """

        original_nbytes = X.nbytes
        U_compressed, S_compressed, V_compressed = self.compress(U, S, V, k)
        compressed_nbytes = (
            U_compressed.nbytes + S_compressed.nbytes + V_compressed.nbytes
        )
        savings = original_nbytes - compressed_nbytes

        return original_nbytes, compressed_nbytes, savings

    def nbytes_to_string(self, nbytes: int, ndigits: int = 3) -> str:
        """
        PROVIDED TO STUDENTS

        Helper function to convert number of bytes to a readable string

        Args:
            nbytes (int): number of bytes
            ndigits (int): number of digits to round to

        Returns:
            str: string representing the number of bytes
        """
        if nbytes == 0:
            return "0B"
        units = ["B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]
        scale = 1024
        units_idx = 0
        n = nbytes
        while n > scale:
            n = n / scale
            units_idx += 1
        return f"{round(n, ndigits)} {units[units_idx]}"

