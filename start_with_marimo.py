import marimo

__generated_with = "0.10.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from marimo_funcs import mo, create_project, options, runButton, start

    mo.md("# GAN")
    return create_project, mo, options, runButton, start


@app.cell
def _(create_project):
    create_project
    return


@app.cell
def _(options):
    options
    return


@app.cell
def _(runButton):
    runButton
    return


@app.cell
def _(mo, runButton, start):
    mo.stop(not runButton.value)

    print('Running...')
    start()
    print('Done!')
    return


if __name__ == "__main__":
    app.run()
