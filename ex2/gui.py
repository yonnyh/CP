import tkinter as tk
from PIL import ImageTk, Image

COL_INDEX = 0
ANGLE_INDEX = 1
FRAME_INDEX = 2




def view_point():
    def update_text_box(val, index):
        text_box_vars[index].set(val)

    def update_slider(index):
        try:
            sliders[index].set(float(text_box_vars[index].get()))
        except ValueError:
            pass

    root = tk.Tk()
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)
    root.title("Viewpoint GUI")
    sliders = []
    text_box_vars = [tk.StringVar(), tk.StringVar(), tk.StringVar()]

    # input frame
    tk.Label(root, text='Get input').grid(row=0, column=0, sticky='ew')
    input_frame = tk.Frame(root, relief='raised', borderwidth=3)
    tk.Checkbutton(input_frame, text='Make transfer motion', command=None).pack(side='right', padx=10, pady=10)
    tk.Button(input_frame, text='Load images',  command=None).pack(side='right',padx=60, pady=25)
    tk.Entry(input_frame, bd=5, width=50, command=None).pack(side='left')
    input_frame.grid(row=1, column=0, sticky='ew')

    # controller frame
    tk.Label(root, text='Viewpoint controller').grid(row=2, column=0, sticky='ew')
    controller_frame = tk.Frame(root, relief='raised', borderwidth=3)

    # starting col slider
    tk.Label(controller_frame, text='Starting column:').grid(row=0, column=0, sticky='we')
    col_slider = tk.Scale(controller_frame, orient='horizontal', from_=0, to=10,
                            resolution=1, command=lambda val: update_text_box(val, COL_INDEX),
                          cursor='DOT',
                            sliderlength=20,troughcolor='pink',length=150)
    col_slider.grid(row=1, column=0, sticky='w')
    sliders.append(col_slider)

    #  starting col text box
    col_text_box = tk.Entry(controller_frame, width=10,bd=3, textvariable=text_box_vars[COL_INDEX])
    col_text_box.grid(row=1, column=1, sticky='s')
    text_box_vars[COL_INDEX].trace_add('write', lambda *args: update_slider(COL_INDEX))

    # angle slider
    tk.Label(controller_frame, text='Angle:').grid(row=3, column=0, sticky='we')
    angle_slider = tk.Scale(controller_frame, orient='horizontal', from_=1, to=179,
                          resolution=0.1, command=lambda val: update_text_box(val, ANGLE_INDEX),
                          cursor='DOT',
                          sliderlength=20, troughcolor='red', length=150)
    angle_slider.grid(row=4, column=0, sticky='w')
    sliders.append(angle_slider)

    #  angle text box
    angle_text_box = tk.Entry(controller_frame, width=10, bd=3, textvariable=text_box_vars[ANGLE_INDEX])
    angle_text_box.grid(row=4, column=1, sticky='s')
    text_box_vars[ANGLE_INDEX].trace_add('write', lambda *args: update_slider(ANGLE_INDEX))

    # starting frame slider
    tk.Label(controller_frame, text='Starting frame:').grid(row=6, column=0, sticky='we')
    frame_slider = tk.Scale(controller_frame, orient='horizontal', from_=0, to=20,
                            resolution=1, command=lambda val: update_text_box(val, FRAME_INDEX),
                            cursor='DOT',
                            sliderlength=20, troughcolor='blue', length=150)
    frame_slider.grid(row=7, column=0, sticky='w')
    sliders.append(frame_slider)

    #  starting frame text box
    frame_text_box = tk.Entry(controller_frame, width=10, bd=3, textvariable=text_box_vars[FRAME_INDEX])
    frame_text_box.grid(row=7, column=1, sticky='s')
    text_box_vars[FRAME_INDEX].trace_add('write', lambda *args: update_slider(FRAME_INDEX))

    #  starting frame canvas TODO

    #groove, raised, ridge, solid, or sunken
    # img = ImageTk.PhotoImage(Image.open("data/Banana/rsz_capture_00001.jpg"))
    # canvas_frame = tk.Label(controller_frame, image=img,  highlightthickness=0, borderwidth=0)
    # canvas_frame.grid(row=2, column=2, sticky='nsew')


    controller_frame.grid(row=2, column=0, sticky='ew')


    root.mainloop()


if __name__ == '__main__':
    view_point()