#!/usr/bin/env python
            # draw the multiplot
            host = host_subplot(111, axes_class=AA.Axes)
            plt.subplots_adjust(right=0.75)
            par1 = host.twinx()
            par2 = host.twinx()
            par3 = host.twinx()
            par4 = host.twinx()
            offset = 50
            fixed_axis_2 = par2.get_grid_helper().new_fixed_axis
            fixed_axis_3 = par3.get_grid_helper().new_fixed_axis
            fixed_axis_4 = par4.get_grid_helper().new_fixed_axis

            par1.axis["right"] = fixed_axis_2(loc="right", axes=par2,offset=(offset, 0))
            par1.axis["right"].toggle(all=True)
            par2.axis["right"] = fixed_axis_3(loc="right", axes=par3,offset=(offset*2, 0))
            par2.axis["right"].toggle(all=True)
            par3.axis["right"] = fixed_axis_4(loc="right", axes=par4,offset=(offset*3, 0))
            par3.axis["right"].toggle(all=True)

            # host.set_xlim(0, 2)
            # host.set_ylim(0, 2)
            host.set_xlabel("Radius [cm]")
            host.set_ylabel("Density")

            par1.set_ylabel("Temperature [K]")
            par2.set_ylabel("Escape Velocity")
            par3.set_ylabel("Gravitational Potential")

            p1, = host.plot(r, rho_sample, label="Density", color=palette1(5/10), linewidth=2)
            p2, = par1.plot(r, T_sample, label="Temperature", color=palette(3/10), linewidth=2)
            p3, = par2.plot(r, v_esc_sample, label="Escape Velocity", color=palette(6/10), linewidth=2)
            p4, = par3.plot(r, phi_sample, label="Gravitational Potential", color=palette(8.6/10), linewidth=2)

            b1, = host.plot(r_poly, rho_poly_sample, label="N=3", color=palette1(5/10), linewidth=2, ls='--')
            b2, = par1.plot(r_poly, T_poly_sample, label="N=3", color=palette(3/10), linewidth=2, ls='--')
            b3, = par2.plot(r_poly, v_esc_poly_sample, label="N=3", color=palette(6/10), linewidth=2, ls='--')
            b4, = par3.plot(r_poly, phi_poly_sample, label="N=3", color=palette(8.6/10), linewidth=2, ls='--')

            # par1.set_ylim(0, 4)
            # par2.set_ylim(1, 65)
            host.tick_params(axis='y', colors=p1.get_color())
            par1.tick_params(axis='y', colors=p2.get_color())
            par2.tick_params(axis='y', colors=p3.get_color())
            par3.tick_params(axis='y', colors=p4.get_color())

            host.legend()
            # host.axis["left"].label.set_color(p1.get_color())
            # par1.axis["right"].label.set_color(p2.get_color())
            # par2.axis["right"].label.set_color(p3.get_color())
            # par3.axis["right"].label.set_color(p4.get_color())
            # par4.axis["right"].label.set_color(p5.get_color())
            plt.draw()
            plt.show()

