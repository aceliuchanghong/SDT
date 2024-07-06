import pickle

if __name__ == '__main__':
    pkl_path = r'D:\aProject\py\SDT\z_new_start\generate_utils\new_character_dict.pkl'
    char_dict = pickle.load(open(pkl_path, 'rb'))
    print(char_dict)
