"""Export Figures 1-9 as RGB TIFF (LZW, 300 dpi; width capped at 180 mm by raising dpi when the pixel width exceeds 2126)."""
from PIL import Image
# manuscript figure number -> source PNG (figures are numbered in order of first mention in the text)
names = {1: 'fig1_pipeline', 2: 'fig2_prisma', 3: 'fig3_corpus', 4: 'fig4_countries', 5: 'fig5_coword', 6: 'fig6_selection', 7: 'fig7_landscape', 8: 'fig8_trends', 9: 'fig9_domains'}
for k, n in names.items():
    im = Image.open(f'figs/{n}.png')
    if im.mode in ('RGBA', 'LA', 'P'):
        bg = Image.new('RGB', im.size, (255, 255, 255)); bg.paste(im.convert('RGBA'), mask=im.convert('RGBA').split()[-1]); im = bg
    else:
        im = im.convert('RGB')
    w, h = im.size
    dpi = max(300, w / (180 / 25.4))  # keep final width <= 180 mm
    im.save(f'tiff/Figure_{k}.tif', dpi=(dpi, dpi), compression='tiff_lzw')
    print(f'Figure_{k}.tif', im.mode, im.size, f'{dpi:.0f} dpi', f'{w / dpi * 25.4:.0f} x {h / dpi * 25.4:.0f} mm')
