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
    start()
    print('Done!')
    return


if __name__ == "__main__":
    app.run()
