import marimo

__generated_with = "0.9.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def __():
    from marimo_funcs import mo, create_project, options, runButton, start

    mo.md("# GAN")
    return create_project, mo, options, runButton, start


@app.cell(hide_code=True)
def __(create_project):
    create_project
    return


@app.cell(hide_code=True)
def __(options):
    options
    return


@app.cell(hide_code=True)
def __(runButton):
    runButton
    return


@app.cell(hide_code=True)
def __(mo, runButton, start):
    mo.stop(not runButton.value)

    print('Running...')
    fig_comp, fig_peaks, fig_means = start()
    print('Done!')
    return fig_comp, fig_means, fig_peaks


@app.cell(hide_code=True)
def __(mo):
    mo.md("""# Plots""")
    return


@app.cell(hide_code=True)
def __(fig_comp, mo):
    mo.mpl.interactive(fig_comp)
    return


@app.cell(hide_code=True)
def __(fig_peaks, mo):
    mo.mpl.interactive(fig_peaks)
    return


@app.cell(hide_code=True)
def __(fig_means, mo):
    mo.mpl.interactive(fig_means)
    return


if __name__ == "__main__":
    app.run()
