from tkinter import *
from skimage.io import imread, imsave
import sys
import numpy as np

FOCUS_GUI = 0
VIEWPOINT_GUI = 1

FOCUS_SLIDER = 0
FOCUS_RESOLUTION = 1
COL_SLIDER = 0
ANGLE_SLIDER = 1
FRAME_SLIDER = 2


class GUI:
    def __init__(self, gui_number):
        self.root = Tk()
        self.gui_number = gui_number

        # self.parent.resizable(False, False)
        self.text_box_vars = [StringVar() for _ in range(self.gui_number + 2)]
        # self.slider_vars = [str() for _ in range(self.gui_number * 2 + 1)]
        self.sliders = []

        # input
        Label(self.root, text='Input').grid(row=0, column=0, sticky='nw')
        self.__init_input_frame()
        self.frame_input.grid(row=1, column=0, sticky='nwe')

        # tools
        if self.gui_number == FOCUS_GUI:
            self.root.title("Focus changer")
            tools_label_text = 'Focus controls'
            self.__init_focus_tools_frame()
        else:
            self.root.title("Viewpoint changer")
            tools_label_text = 'Viewpoint controls'
            self.__init_viewpoint_tools_frame()
        Label(self.root, text=tools_label_text).grid(row=3, column=0, sticky='nw')
        self.frame_tools.grid(row=4, column=0, sticky='new')

        # output
        Label(self.root, text='Output').grid(row=0, column=1, sticky='nw')
        self.__init_output_frame()
        self.frame_output.grid(row=1, column=1, rowspan=5, sticky='nsew')

        self.root.grid_rowconfigure(4, weight=1)
        self.root.grid_columnconfigure(0, minsize=300, weight=0)
        self.root.grid_columnconfigure(1, weight=1)

        self.init_textbox_values()
        self.root.mainloop()

    def __init_output_frame(self):
        self.frame_output = Frame(self.root, bg='blue')

        # self.label_output_image = Label(self.frame_output, image=self.output_image_preview)
        # self.label_output_image.image = self.output_image_preview
        # self.label_output_image.grid(row=1, column=0)

    def __init_input_frame(self):
        # frame
        self.frame_input = Frame(self.root, relief=RAISED, borderwidth=2)

        # load images button
        self.button_load_images = Button(self.frame_input, text='Load images...', command=None)
        self.button_load_images.pack(padx=10, pady=10)

    def __init_focus_tools_frame(self):
        # frame
        self.frame_tools = Frame(self.root, relief=RAISED, borderwidth=2)

        # focus slider
        Label(self.frame_tools, text='Focus:').grid(row=0, column=0, sticky=W)
        self.sliders.append(Scale(self.frame_tools, orient=HORIZONTAL, from_=.1, to=3, resolution=0.1, length=200,
                                  command=lambda new_val: self.update_textbox(new_val, FOCUS_SLIDER), showvalue=0))
        self.sliders[FOCUS_SLIDER].grid(row=0, column=1, columnspan=2)

        # manual number textbox
        self.focus_textbox = Entry(self.frame_tools, width=6, textvariable=self.text_box_vars[FOCUS_SLIDER])
        self.focus_textbox.grid(row=0, column=3, sticky=S)
        self.text_box_vars[FOCUS_SLIDER].trace_add('write', lambda *args: self.update_slider_from_textbox(FOCUS_SLIDER))

        # slider resolution
        Label(self.frame_tools, text='Slider resolution:').grid(row=1, column=1, sticky=E)
        self.slider_resolution_textbox = Entry(self.frame_tools, width=6, textvariable=self.text_box_vars[FOCUS_RESOLUTION])
        self.slider_resolution_textbox.grid(row=1, column=2, sticky=W)
        self.text_box_vars[FOCUS_RESOLUTION].trace_add('write', lambda *args: self.update_focus_resolution())

        # mark button
        Button(self.frame_tools, text='Mark point manually...', command=None).grid(row=2, column=1, columnspan=2)

        # remove occlusions checkbox
        self.occlusions_checkbox = Checkbutton(self.frame_tools, text="Remove occlusions", variable=None)
        self.occlusions_checkbox.grid(row=3, column=0, columnspan=3, sticky=W)

    def __init_viewpoint_tools_frame(self):
        # frame
        self.frame_tools = Frame(self.root, relief=RAISED, borderwidth=2)

        # start column
        Label(self.frame_tools, text='Starting column:').grid(row=0, column=0, sticky=W+S)
        self.sliders.append(Scale(self.frame_tools, orient=HORIZONTAL, from_=.1, to=3, resolution=0.1, showvalue=0,
                                  command=lambda new_val: self.update_textbox(new_val, COL_SLIDER)))  # , length=250)
        self.sliders[COL_SLIDER].grid(row=1, column=0, sticky=W)
        self.col_textbox = Entry(self.frame_tools, width=6, textvariable=self.text_box_vars[COL_SLIDER])
        self.col_textbox.grid(row=1, column=1, sticky=W)
        self.text_box_vars[COL_SLIDER].trace_add('write', lambda *args: self.update_slider_from_textbox(COL_SLIDER))

        # angle
        Label(self.frame_tools, text='Angle:').grid(row=2, column=0, sticky=W+S)
        self.sliders.append(Scale(self.frame_tools, orient=HORIZONTAL, from_=.1, to=3, resolution=0.1, showvalue=0,
                                  command=lambda new_val: self.update_textbox(new_val, ANGLE_SLIDER)))  # , length=250)
        self.sliders[ANGLE_SLIDER].grid(row=3, column=0, sticky=W)
        self.angle_textbox = Entry(self.frame_tools, width=6, textvariable=self.text_box_vars[ANGLE_SLIDER])
        self.angle_textbox.grid(row=3, column=1, sticky=W)
        self.text_box_vars[ANGLE_SLIDER].trace_add('write', lambda *args: self.update_slider_from_textbox(ANGLE_SLIDER))

        # viewpoint illustration
        self.my_frame_red = Frame(self.frame_tools, bg='red', width=100)
        self.my_frame_red.grid(row=1, column=2, rowspan=3, sticky='nsew')

        # start frame
        Label(self.frame_tools, text='Starting\nframe:').grid(row=0, column=3)
        self.sliders.append(Scale(self.frame_tools, from_=.1, to=3, resolution=0.1, showvalue=0,
                                  command=lambda new_val: self.update_textbox(new_val, FRAME_SLIDER)))  # , length=250)
        self.sliders[FRAME_SLIDER].grid(row=1, column=3, rowspan=3)
        self.frame_textbox = Entry(self.frame_tools, width=6, textvariable=self.text_box_vars[FRAME_SLIDER])
        self.frame_textbox.grid(row=4, column=3)
        self.text_box_vars[FRAME_SLIDER].trace_add('write', lambda *args: self.update_slider_from_textbox(FRAME_SLIDER))

    def update_slider_from_textbox(self, slider_number):
        textbox_value = self.text_box_vars[slider_number].get()
        try:
            value = float(textbox_value)
            self.sliders[slider_number].set(value)
            if value != self.sliders[slider_number].get():
                self.update_textbox(self.sliders[slider_number].get(), slider_number)
        except ValueError:
            pass

    def update_focus_resolution(self):
        textbox_value = self.text_box_vars[FOCUS_RESOLUTION].get()
        try:
            value = float(textbox_value)
            self.sliders[FOCUS_SLIDER]['resolution'] = value
            # if value != self.sliders[FOCUS_SLIDER]['resolution']:
            #     self.update_textbox(self.sliders[FOCUS_RESOLUTION].get(), FOCUS_RESOLUTION)
        except ValueError:
            pass

    def update_textbox(self, new_val, textbox_number):
        self.text_box_vars[textbox_number].set(new_val)

    def init_textbox_values(self):
        for i, slider in enumerate(self.sliders):
            self.update_textbox(slider.get(), i)

        if self.gui_number == FOCUS_GUI:
            self.text_box_vars[FOCUS_RESOLUTION].set(self.sliders[FOCUS_SLIDER]['resolution'])

    def update(self):
        pass

    def apply_tools(self):
        gamma = float(self.focus_scale.get())
        brightness = float(self.scale_brightness.get())

        self.full_output_image = self.orig_output_image ** (1 / gamma)
        self.full_output_image = brightness * self.full_output_image
        self.full_output_image = np.clip(self.full_output_image, 0, 1)

        self.update_preview(self.full_output_image)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        GUI(int(sys.argv[1]))
    else:
        # GUI(FOCUS_GUI)
        GUI(VIEWPOINT_GUI)

