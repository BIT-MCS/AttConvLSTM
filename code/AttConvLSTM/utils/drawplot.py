import matplotlib.pyplot as plt
class Drawplot(object):
    def __init__(self,win_width=5,win_height=5,xlabel=None,ylabel=None,is_draw=True,pause_time=0.01,num=1,row=1,col=1):
        self.myfig=plt.figure(figsize=[win_width,win_height])
        self.ax=[None]*num
        for i in range(num):
            self.ax[i] = self.myfig.add_subplot(row, col, i+1)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if is_draw:
            plt.ion()
            plt.show()

        self.pause_time=pause_time
        self.is_draw=is_draw

    def show_step(self,current_total_reward,current_step,num=0,draw=True):

        if not draw:
            return
        if not self.is_draw:
            plt.ion()
            plt.show()
            self.is_draw=True
        self.ax[num-1].plot(current_step,current_total_reward)
        self.ax[num-1].scatter(current_step, current_total_reward)

        plt.pause(self.pause_time)

    def show_img(self, x,num=1,draw=True):
        if not draw:
            return
        if not self.is_draw:
            plt.ion()
            plt.show()
            self.is_draw=True

        self.ax[num-1].imshow(x)

        plt.pause(self.pause_time)