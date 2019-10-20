import tkinter as tk

root = tk.Tk()
canvas = tk.Canvas(root, height=700, width=800, bg="#e6e6ff")
canvas.pack()

frame1 = tk.Frame(root, bg="#e6e6ff", bd=5)
frame1.place(relx=0.4, rely=0.1, relwidth=0.9, relheight=0.2, anchor='n')

label1 = tk.Label(frame1, text="user_id:", bg="#e6e6ff", justify="left", bd=7)
label1.place(relx=0.1, rely=0.04, relwidth=0.15, relheight=0.3)
entry1 = tk.Entry(frame1, bg="#f2f2f2")
entry1.place(relx=0.3, rely=0.04, relwidth=0.7, relheight=0.3)

label2 = tk.Label(frame1, text="Anime Name:", bg="#e6e6ff", justify="left", bd=7)
label2.place(relx=0.1, rely=0.36, relwidth=0.15, relheight=0.3)
entry2 = tk.Entry(frame1, bg="#f2f2f2")
entry2.place(relx=0.3, rely=0.36, relwidth=0.7, relheight=0.3)

label3 = tk.Label(frame1, text="Rating:", bg="#e6e6ff", justify="left", bd=7)
label3.place(relx=0.1, rely=0.68, relwidth=0.15, relheight=0.3)
entry3 = tk.Entry(frame1, bg="#f2f2f2")
entry3.place(relx=0.3, rely=0.68, relwidth=0.7, relheight=0.3)

button = tk.Button(root, text="Find the similer anime", bg="#6600ff", activebackground="#1f1f7a", bd=5)
button.place(relx=0.55, rely=0.3, relwidth=0.2, relheight=0.05, anchor='n')

root.mainloop()