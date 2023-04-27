class Person:
    """
    Представляет загруженное в память изображение 
    человека и его метку
    """

    def __init__(self, image, id) -> None:
        self.image = image
        self.id = id
