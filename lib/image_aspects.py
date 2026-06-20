from enum import Enum


class ImageAspects(Enum):
    SQUARE = 1.
    POSTCARD = 2 / 3.
    POSTCARD_INVERSE = 3 / 2.
    FIVE_FOUR = 5 / 4.
    FIVE_FOUR_INVERSE = 4. / 5

    def code(self):
        _code = {
            ImageAspects.SQUARE: "1:1",
            ImageAspects.POSTCARD: "3/2",
            ImageAspects.POSTCARD_INVERSE: "2/3",
            ImageAspects.FIVE_FOUR: "5/4",
            ImageAspects.FIVE_FOUR_INVERSE: "4/5"
        }
        return _code.get(self, "Unknown aspect")

    @classmethod
    def from_code(cls, code):
        _code = {
            "1:1": ImageAspects.SQUARE,
            "3/2": ImageAspects.POSTCARD,
            "2/3": ImageAspects.POSTCARD_INVERSE,
            "5/4": ImageAspects.FIVE_FOUR,
            "4/5": ImageAspects.FIVE_FOUR_INVERSE
        }
        return _code.get(code, ImageAspects.SQUARE)
