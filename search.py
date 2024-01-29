import os
import argparse
import pickle

from DeepImageSearch import Load_Data, Search_Setup


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep Image Search')
    parser.add_argument('--dir', type=str, default='images', required=True)
    parser.add_argument('--model', type=str,
                        default='vit_base_patch16_224_in21k')
    parser.add_argument('--num', type=int, default=10)
    parser.add_argument('--query', type=str, required=True)

    args = parser.parse_args()
    image_list = Load_Data().from_folder([args.dir])
    st = Search_Setup(image_list=image_list,
                      model_name=args.model, pretrained=True)

    # Find existing index
    existing_images = []
    if os.path.exists(f'metadata-files/{args.model}'):
        with open(f'metadata-files/{args.model}/image_data_features.pkl', 'rb') as f:
            d = pickle.load(f)

        existing_images = d.images_paths

    st.add_images_to_index(
        list(set(image_list).difference(set(existing_images))))

    results = st.get_similar_images(args.query, args.num)
    print(list(results.values()))
