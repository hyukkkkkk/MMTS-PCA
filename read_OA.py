import numpy as np

def read_ortho_arr(file_path):

    ortho_array = []
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()  # 줄 끝의 줄 바꿈 문자를 제거한다.
            line = [int(t) for t in line]
            ortho_array.append(line)
    ortho_array = np.array(ortho_array)
    
    return ortho_array



