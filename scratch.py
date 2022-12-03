# from data import o_pay_1922_1942, e_pay_1922_1942

# # data structure {year -> (rank, dict(time in service: blank)}}


# o_pay = {}
# for year in range(1922, 1943):
#     temp_dict = {}
#     for d in o_pay_1922_1942:
#         for rank, d2 in d.items():
#             temp_dict[rank] = {**temp_dict.get(rank, {}), **d2}
#     o_pay[year] = temp_dict

# import pprint

# pprint.pprint(o_pay)

from tabula import read_pdf
from tabulate import tabulate
import os

import layoutparser as lp
import cv2
import numpy as np

pdf_path = os.path.join("./pay_tables", "MilPayTable1949.pdf")

from pdf2image import convert_from_path

pages = convert_from_path(pdf_path)
image = np.array(pages[0])
# image = cv2.imread(img)
# print(img)
image = image[..., ::-1]

ocr_agent = lp.TesseractAgent()
res = ocr_agent.detect(image, return_response=True)

layout = ocr_agent.gather_data(res, agg_level=lp.TesseractFeatureType.WORD)

import matplotlib.pyplot as plt

lp.draw_text(image, layout, font_size=12, with_box_on_text=True, text_box_width=1)

# layout = ocr_agent.gather_full_text_annotation(res, agg_level=lp.GCVFeatureType.WORD)

# # collect all the layout elements of the `WORD` level
