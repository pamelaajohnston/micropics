# Program to explain how to create drawing App in kivy

# import kivy module
import kivy

# base Class of your App inherits from the App class.
# app:always refers to the instance of your application
from kivy.app import App

# this restrict the kivy version i.e
# below this kivy version you cannot
# use the app or software
kivy.require('1.9.0')

# Widgets are elements of a
# graphical user interface that
# form part of the User Experience.
from kivy.uix.widget import Widget

# This layout allows you to set relative coordinates for children.
from kivy.uix.relativelayout import RelativeLayout


# Create the Widget class
class Paint_brush(Widget):
    pass


# Create the layout class
# where you are defining the working of
# Paint_brush() class
class Drawing(RelativeLayout):

    # On mouse press how Paint_brush behave
    def on_touch_down(self, touch):
        pb = Paint_brush()
        pb.center = touch.pos
        self.add_widget(pb)

    # On mouse movement how Paint_brush behave
    def on_touch_move(self, touch):
        pb = Paint_brush()
        pb.center = touch.pos
        self.add_widget(pb)


# Create the App class
class DrawingApp(App):
    def build(self):
        return Drawing()


DrawingApp().run()

