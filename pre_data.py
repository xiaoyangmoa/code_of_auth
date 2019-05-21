from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import time

def rnd_char():
    '''
    随机一个字母或者数字
    :return:
    '''
    # 随机一个字母或者数字
    i = random.randint(1,3)
    if i == 1:
        # 随机个数字的十进制ASCII码
        an = random.randint(97, 122)
    elif i == 2:
        # 随机个小写字母的十进制ASCII码
        an = random.randint(65, 90)
    else:
        # 随机个大写字母的十进制ASCII码
        an = random.randint(48, 57)
    # 根据Ascii码转成字符，return回去
    return chr(an)

def rnd_color2():
    '''
      随机颜色，规定一定范围
      :return:
      '''
    return (random.randint(32, 127), random.randint(32, 127), random.randint(32, 127))

def rnd_color():
    '''
    随机颜色，规定一定范围
    :return: 
    '''
    return (random.randint(64, 255), random.randint(64, 255), random.randint(64, 255))

def create_code():
    # 240 x 60:
    width = 60 * 4
    height = 60
    image = Image.new('RGB', (width, height), (192, 192, 192))
    # 创建Font对象:
    font = ImageFont.truetype(r'E:\pycharm\code_of_auth\font\simfang.ttf',36)
    # 创建Draw对象:
    draw = ImageDraw.Draw(image)

    # 填充每个像素:
    for x in range(0, width, 20):
        for y in range(0, height, 10):
            draw.point((x, y), fill=rnd_color())

    # 填充字符
    _str = ""
    # 填入4个随机的数字或字母作为验证码
    for t in range(4):
        c = rnd_char()
        _str = "{}{}".format(_str, c)
        # 随机距离图片上边高度，但至少距离30像素
        h = random.randint(1, height - 30)
        # 宽度的化，每个字符占图片宽度1／4,在加上10个像素空隙
        w = width / 4 * t + 10
        draw.text((w, h), c, font=font, fill=rnd_color2())
        # 实际项目中，会将验证码 保存在数据库，并加上时间字段
        print("验证码生成完毕{}".format(_str))
    t = time.time()
    current_time = int(round(t * 1000))
    save_dir = 'train_data/auth_code_{}.jpg'.format(current_time)
    image.save(save_dir, 'jpeg')

for i in range(100):
    create_code()
