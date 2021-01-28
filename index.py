import os
import predict
from tornado.web import RequestHandler


class MainHandler(RequestHandler):
    def get(self, *args, **kwargs):
        images = self.get_argument("imgs")
        image_list = images.split(",")
        res = predict.test(image_list)
        data = {
            "status": 200,
            "res": res
        }
        self.write(data)

    def post(self, *args, **kwargs):
        file_img = self.request.files.get('file')
        if not os.path.exists("img"):
            os.mkdir("img")
        for img in file_img:
            with open("img/" + img.filename, 'wb') as f:
                f.write(img.body)

        self.write("200")
