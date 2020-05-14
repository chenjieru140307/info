# qrcode

二维码：

1. 可存储的信息量大：可容纳多达 1850 个大写字母或 2710 个数字或 1108 个字节或 500 多个汉字。
2. 容错能力强：具有纠错功能，这使得二维条码因穿孔、污损等引起局部损坏时，照样可以正确得到识读，损毁面积达 30% 仍可恢复信息。
3. 译码可靠性高：它比普通条码译码错误率百万分之二要低得多，误码率不超过千万分之一。
4. 激光可识别。

比如我们现在常见的公众号二维码，他就利用了二维码容错能力强的特点，在二维码中间加入了公众号的图标。虽然中间的图片遮盖了一部分二维码的真实数据，但因为其强大的容错能力，所以并没有影响二维码要传递的数据。

<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200515/obmgkUOQ7Dvg.png?imageslim">
</p>



qrcode：

- 二维码生成器，支持生成 GIF 动态、图片二维码。

文档：


- [github](https://github.com/lincolnloop/python-qrcode)
- [用qrcode模块生成二维码](https://blog.csdn.net/jy692405180/article/details/65937077)

安装：

- `pip install qrcode`
- 可能需要升级 pip `pip install --upgrade pip`


举例：

```py
import qrcode
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=10,
    border=4,
)
qr.add_data('http://www.iterate.site')
qr.make(fit=True)

img = qr.make_image(fill_color="black", back_color="white")
img.get_image().show()
img.save('qrcode.png')
```

输出：

<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200515/LoFbv0eh7VB2.png?imageslim">
</p>


