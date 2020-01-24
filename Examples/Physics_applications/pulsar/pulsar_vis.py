import yt
import os
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--plotfile", type=str, help="Path to single input plotfile to visualize.")
parser.add_argument("-d", "--dir", type=str, help="Path to input plotfile directory to visualize.")

args = parser.parse_args()

def visualize(plt, annotate_particles=True):
    ds = yt.load(plt)
    sl = yt.SlicePlot(ds, 0, 'Ez', aspect=1) # Create a sliceplot object
    if annotate_particles:
        sl.annotate_particles(width=(2.e3, 'm'),ptype="plasma_p",col='b',p_size=5.0)
        sl.annotate_particles(width=(2.e3, 'm'),ptype="plasma_e",col='r',p_size=5.0)
    sl.annotate_streamlines("Ey", "Ez", plot_args={"color": "black"})
    sl.annotate_grids() # Show grids
    sl.annotate_sphere([90000.0, 90000.0, 90000.0], radius=12000.0,
                              circle_args={'color':'white', 'linewidth':2})
    out_name = os.path.join(os.path.dirname(plt), "pulsar_{}.png".format(os.path.basename(plt)))
    sl.save(out_name)

if __name__ == "__main__":
    yt.funcs.mylog.setLevel(50)
    if args.dir:
        failed = []
        plotfiles = glob.glob(os.path.join(args.dir, "plt" + "[0-9]"*5))
        for pf in plotfiles:
            try:
                visualize(pf)
            except:
                # plotfile 0 may not have particles, so turn off annotations if we first failed
                try:
                    visualize(pf, annotate_particles=False)
                except:
                    failed.append(pf)
                    pass

        if len(failed) > 0:
            print("Visualization failed for the following plotfiles:")
            for fp in failed:
                print(fp)
        else:
            print("No plots failed, creating a gif ...")
            input_files = os.path.join(args.dir, "*.png")
            output_gif  = os.path.join(args.dir, "pulsar.gif")
            os.system("convert -delay 20 -loop 0 {} {}".format(input_files, output_gif))

    elif args.plotfile:
        visualize(args.plotfile)
    else:
        print("Supply either -f or -d options for visualization. Use the -h option to see help.")
