import time
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# For referencing global price data if needed
from trading import price_history_upbit, price_history_binance, volume_history_upbit, volume_history_binance

enable_plotting = False
plot_window = None
fig = None
ax1 = None
ax2 = None
ax3 = None  # if you’re using a second twin axis

def open_plot_window():
    global plot_window, fig, ax1, ax2
    if plot_window is not None:
        return
    plot_window = tk.Toplevel()
    plot_window.title("Live BTC Chart")

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    canvas = FigureCanvasTkAgg(fig, master=plot_window)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def close_plot_window():
    global plot_window, fig, ax1, ax2
    if plot_window is not None:
        plot_window.destroy()
        plot_window = None
        fig = None
        ax1 = None
        ax2 = None

def main_control_ui():
    def toggle_plot_window():
        nonlocal btn
        global enable_plotting
        enable_plotting = not enable_plotting
        btn.config(text="Close Plot Window" if enable_plotting else "Open Plot Window")
        if enable_plotting:
            open_plot_window()
            schedule_plot_update()  # Start updates now
        else:
            close_plot_window()

    root = tk.Tk()
    root.title("Main Control Board")
    btn = tk.Button(root, text="Open Plot Window", command=toggle_plot_window)
    btn.pack(padx=20, pady=20)
    root.mainloop()

def schedule_plot_update():
    """
    Schedule the next UI update after 2000 ms (2 sec)
    """
    if enable_plotting and plot_window is not None and ax1 is not None:
        update_plot()
    # Re-schedule another update in 2000 ms
    if plot_window is not None:  # Still open?
        plot_window.after(2000, schedule_plot_update)

def update_plot():
    """
    The actual plotting logic. Called by schedule_plot_update (on main thread).
    """
    ax1.clear()
    ax2.clear()

    ax1.set_title("Live BTC Price & Volume (Upbit / Binance)")
    ax1.set_ylabel("Price (KRW)")

    up_len = len(price_history_upbit)
    bn_len = len(price_history_binance)
    comm_len_price = min(up_len, bn_len)

    ax1.plot(range(comm_len_price),
             price_history_upbit[:comm_len_price],
             color='blue', label='Upbit BTC')
    ax1.plot(range(comm_len_price),
             price_history_binance[:comm_len_price],
             color='red', label='Binance BTC')
    ax1.legend(loc='upper left')

    # If you want separate axis for binance volume
    # ax3 = ax1.twinx()  # etc.

    up_vol_len = len(volume_history_upbit)
    bn_vol_len = len(volume_history_binance)
    comm_len_vol = min(up_vol_len, bn_vol_len)

    x_up = range(comm_len_vol)
    volume_up = volume_history_upbit[:comm_len_vol]
    volume_bn = volume_history_binance[:comm_len_vol]

    print("Debug - Upbit volume:", volume_history_upbit, len(volume_history_upbit))  # Debug print

    ax2.set_ylabel("Volume over Time")

    # Replace bar plots with line plots for volume
    ax2.plot(range(comm_len_vol), volume_up, color='blue', marker='o', label='Upbit Volume')
    ax2.plot(range(comm_len_vol), volume_bn, color='red',  marker='o', label='Binance Volume')

    # Remove or adjust the ylim if desired
    ax2.set_ylim([0, max(1, max(volume_up + volume_bn, default=0)) * 1.2])

    ax2.legend(loc='upper right')

    fig.canvas.draw()