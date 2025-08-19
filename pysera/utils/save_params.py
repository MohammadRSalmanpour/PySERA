"""
Functions for writing parameters on excel file
"""

from typing import Dict, Any
import os
from openpyxl import load_workbook, Workbook


def write_to_excel(
        excel_path: str,
        # RoI_name: str, add later
        all_params: list[tuple],
        sheet2write: str) -> None:
    PARAM_SHEET_HEADERS = ["apply_preprocessing", "min_roi_volume",
                           "feat2out", "roi_num", "roi_selection_mode"]

    for i, params in enumerate(all_params):  # takes only 1 - modify if required
        img_path = params[0]
        mask_input = params[1]  # will handle later
        hparams = params[2]  # dict
        output_path = params[3]  # duplicated in hyper_params
        apply_preprocessing = params[4]
        min_roi_volume = params[5]
        folder_name = params[6]  # unused
        feat2out = params[7]
        roi_num = params[8]
        roi_selection_mode = params[9]

        # get patient id
        patient_id = os.path.basename(img_path)

        # Make headers
        if i == 0:
            hparam_headers = hparams.keys()
            PARAM_SHEET_HEADERS.extend(list(hparam_headers))

        # get all values to write on excell
        all_values = [apply_preprocessing, min_roi_volume,
                      feat2out, roi_num, roi_selection_mode] + get_dict_value(hparams)

        clean_headers = pretty_headers(PARAM_SHEET_HEADERS)  # removes unwanted parts of headers

        if not os.path.exists(excel_path):
            wb = Workbook()
            ws = wb.active
            ws.title = sheet2write
            ws.append(clean_headers)
            ws.append(all_values)
            wb.save(excel_path)
        else:
            wb = load_workbook(excel_path)
            if sheet2write not in wb.sheetnames:
                ws = wb.create_sheet(title=sheet2write)
                ws.append(clean_headers)
            else:
                ws = wb[sheet2write]

            ws.append(all_values)
            wb.save(excel_path)


# Used in write_to_excel to get keys and values of parameters' dictionary as headers and values of excel file, respectively.
def get_dict_value(params_dict: Dict[str, Any]) -> list[Any]:
    values = []
    for k, v in params_dict.items():
        values.append(v)

    return values


def pretty_headers(list_of_headers: list[str]) -> list[str]:
    return [h.replace("radiomics_", "") for h in list_of_headers]
