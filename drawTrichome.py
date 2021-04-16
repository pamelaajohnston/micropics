from tkinter import *
from tkinter.colorchooser import askcolor
from PIL import Image
from tkinter import filedialog
import io

class Paint(object):

    DEFAULT_PEN_SIZE = 10.0
    DEFAULT_COLOR = 'black'
    backgroundColour = '#%02x%02x%02x' % (125, 125, 125)  # set your favourite rgb color
    trichomeColour = '#%02x%02x%02x' % (0, 0, 0)  # set your favourite rgb color
    cellColour = '#%02x%02x%02x' % (255, 255, 255)  # set your favourite rgb color
    dotColour = '#%02x%02x%02x' % (255, 0, 0)  # set your favourite rgb color
    #mycolour2 = '#40E0D0'  # or use hex if you prefer
    myCapstyle = ROUND
    canvasW = 224
    canvasH = 224

    def __init__(self):
        self.root = Tk()

        self.trichome_button = Button(self.root, text='trichome', command=self.draw_trichome)
        self.trichome_button.grid(row=0, column=0)

        self.background_button = Button(self.root, text='background', command=self.draw_background)
        self.background_button.grid(row=0, column=1)

        self.cell_button = Button(self.root, text='cell (centre)', command=self.draw_cell)
        self.cell_button.grid(row=0, column=2)

        self.save_button = Button(self.root, text='save', command=self.save_image)
        self.save_button.grid(row=0, column=3)

        self.reset_button = Button(self.root, text='reset', command=self.reset_background)
        self.reset_button.grid(row=0, column=4)


        #self.choose_size_button = Scale(self.root, from_=1, to=10, orient=HORIZONTAL)
        #self.choose_size_button.grid(row=0, column=4)

        self.c = Canvas(self.root, bg=self.backgroundColour, width=self.canvasW, height=self.canvasH)
        self.c.grid(row=1, columnspan=5)

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = self.DEFAULT_PEN_SIZE
        self.color = self.DEFAULT_COLOR
        self.active_button = self.trichome_button
        self.c.bind('<Button-1>', self.doCell)
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)
        self.reset_background()

    def draw_trichome(self):
        self.activate_button(self.trichome_button)
        self.line_width = 15.0

    def draw_background(self):
        self.activate_button(self.background_button)
        self.line_width = 15.0

    def reset_background(self):
        self.activate_button(self.reset_button)
        self.c.create_rectangle(0,0,(self.canvasW+20),(self.canvasH+20), outline=self.backgroundColour, fill=self.backgroundColour)

    def draw_cell(self):
        self.activate_button(self.cell_button)
        self.line_width = 20.0

    def save_image(self):
        self.activate_button(self.save_button)
        f = filedialog.asksaveasfilename(defaultextension=".png", initialfile="myfile.png", title="Save Canvas")
        if f is None: # asksaveasfile return `None` if dialog closed with "cancel".
            return
        ps = self.c.postscript(colormode='color')
        img = Image.open(io.BytesIO(ps.encode('utf-8')))
        img = img.resize((224, 224))
        img.save(f, 'png')

    #def choose_color(self):
    #    self.eraser_on = False
    #    self.color = askcolor(color=self.color)[1]

    #def use_eraser(self):
    #    self.activate_button(self.eraser_button, eraser_mode=True)

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        #self.eraser_on = eraser_mode

    def paint(self, event):
        #self.line_width = self.choose_size_button.get()
        paint_color = self.trichomeColour
        if self.active_button == self.cell_button:
            paint_color = self.cellColour
            return self.doCell(event)
        if self.active_button == self.background_button:
            paint_color = self.backgroundColour
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=self.myCapstyle, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y

    def doCell(self, event):
        paint_color = self.cellColour
        #self.c.create_oval((event.x-1), (event.y-1), (event.x+1), (event.y+1), outline=paint_color, fill=paint_color, width=2)
        self.c.create_rectangle((event.x-1),(event.y-1), (event.x+1), (event.y+1), outline=paint_color, fill=paint_color)



    def reset(self, event):
        self.old_x, self.old_y = None, None


if __name__ == '__main__':
    Paint()
