from util import create_data_lists
from pathlib import Path

ROOT = Path("/home/famousdeer/Desktop/Praca magisterska/Program/data/VOCdevkit")
VOC2012 = ROOT / "VOC2012"
VOC2007 = ROOT / "VOC2007"

if __name__ == '__main__':
    create_data_lists(voc07_path=VOC2007,
                      voc12_path=VOC2012,
                      output_folder=ROOT)