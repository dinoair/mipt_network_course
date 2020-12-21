from django import forms

# default_face_url = "https://sun9-38.userapi.com/impf/c852232/v852232682/61af3/ptQeoCvkBl4.jpg?size=754x564&quality=96&proxy=1&sign=fe3de753f68373d5a5cc5b02733dc20e"
# default_style_url = "https://sun9-55.userapi.com/impf/609j0tXE1iWvWJlB5RpSCqJFJ6RBsAe3KSKliw/UtaQaieYybg.jpg?size=447x604&quality=96&proxy=1&sign=383fd7a71fe1be901eba4c190b20e5b3"

class UrlsForm(forms.Form):
    portret_url = forms.URLField(label='Portret Url')
    style_url = forms.URLField(label='Background Url')
