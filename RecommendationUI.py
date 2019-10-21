import tkinter as tk
import Find_data as fd

root = tk.Tk()
canvas = tk.Canvas(root, height=700, width=800, bg="#e6e6ff")
canvas.pack()

def split_anime(anime_list):
    st = 'Related Animes:\n'
    for x in anime_list:
        st = st+x+'\n'
    return st

def Get_Related_Anime(name):
    rel_anime = fd.get_reated_anime(name)
    st = split_anime(rel_anime)
    #Cross Game
    frame3 = tk.Frame(root, bg="#e6e6ff", bd=5)
    frame3.place(relx=0.50, rely=0.52, relwidth=0.8, relheight=0.43, anchor='n')

    label_output = tk.Label(frame3, text=st, bg="#e6e6ff", justify="left", bd=7)
    label_output.place(relx=0.1, rely=0.1, relwidth=0.9, relheight=0.9)



def Get_Anime_Genre(id, name, rate):
    g,index = fd.get_genre(name)
    frame2 = tk.Frame(root, bg="#e6e6ff", bd=5)
    frame2.place(relx=0.50, rely=0.36, relwidth=0.8, relheight=0.07, anchor='n')

    label4 = tk.Label(frame2, text="Genre:", bg="#e6e6ff", justify="left", bd=7)
    label4.place(relx=0.1, rely=0.04, relwidth=0.15, relheight=0.9)
    gen_label = tk.Label(frame2, text=g, bg="#f2f2f2")
    gen_label.place(relx=0.3, rely=0.04, relwidth=0.7, relheight=0.9)

    button = tk.Button(root, text="Find the similer anime", bg="#6600ff", activebackground="#1f1f7a", bd=5, command=lambda: Get_Related_Anime(name))
    button.place(relx=0.55, rely=0.45, relwidth=0.2, relheight=0.05, anchor='n')



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

button = tk.Button(root, text="Type of the anime", bg="#6600ff", activebackground="#1f1f7a", bd=5, command=lambda: Get_Anime_Genre(entry1.get(), entry2.get(), entry3.get()))
button.place(relx=0.55, rely=0.3, relwidth=0.2, relheight=0.05, anchor='n')

root.mainloop()