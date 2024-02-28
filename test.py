import argparse
import os
import time

from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm


def main(args: argparse.Namespace):

    # statistic
    total_img_without_error = 0
    error_to_align = 0
    #

    mtcnn = MTCNN(image_size=args.imgsz)
    persons = os.listdir(args.input)
    for person_id, person in tqdm(
        enumerate(persons), total=len(persons), desc="person"
    ):
        img_persons = os.listdir(f"{args.input}/{person}")
        total_img_without_error += len(img_persons)
        for img_id, img_person in tqdm(
            enumerate(img_persons), total=len(img_persons), desc="img_person"
        ):
            input = f"{args.input}/{person}/{img_person}"
            output = (
                f"{args.output}/{person}/{img_id}.jpg"
                if args.anon == False
                else f"{args.output}/{person_id}/{img_id}.jpg"
            )
            try:
                img = Image.open(input)
                mtcnn(img, save_path=output)
            except Exception as e:
                print(f"error while converting {input}")
                error_to_align += 1

    print(f"sucessfull aligned image  = {total_img_without_error - error_to_align}")


if __name__ == "__main__":
    description = "align custom datasets so it match the required format"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input", type=str, help="input folder", required=True)
    parser.add_argument(
        "--imgsz", type=int, help="convert to imgsz x imgsz image", default=112
    )
    parser.add_argument(
        "--anon",
        type=bool,
        help=" anonyimize the input folder (rename each person folder with just number)",
        default=True,
    )
    parser.add_argument("--output", type=str, help="output folder", required=True)
    args = parser.parse_args()
    main(args)
