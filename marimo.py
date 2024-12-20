import marimo

__generated_with = "0.10.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from io import StringIO
    import csv
    import pandas as pd

    from model.params import params
    from main import run
    return StringIO, csv, mo, params, pd, run


@app.cell
def _(StringIO, csv, pd):
    def read_csv(file):
        str_ = str(file.contents(), 'utf-8')
        data = StringIO(str_)
        sep = csv.Sniffer().sniff(data.getvalue()).delimiter
        df = pd.read_csv(data, sep = sep)
        df = df.set_index(df.columns[0])
        return df


    def to_boolean(str_):
        bool_dict = {'yes': True, 'no': False, 'on': True, 'off': False}
        return bool_dict[str_]
    return read_csv, to_boolean


@app.cell
def _(mo, params):
    # Basic tab
    appTitle = mo.md("### **GAN**")
    projectName = mo.ui.text(label = 'Project name:')
    inputFileLabel = mo.md('Input data:')
    inputFile = mo.ui.file(label = 'Upload', filetypes = ['.csv'])
    outputFormat = mo.ui.dropdown(options = ['.npy', '.csv', '.xslx'], value = params['outputFormat'], label = 'Output file format:')
    useWandb = mo.ui.radio(options = ['off', 'on'], value = 'off', inline = True, label = 'Wandb:')
    epochCount = mo.ui.number(start = 1, step = 1, value = params['epochCount'], label = 'Number of epochs:')
    saveFreq = mo.ui.number(start = 1, step = 1, value = params['saveFreq'], label = 'Save frequency ¹:')
    saveSamples = mo.ui.radio(options = ['no', 'yes'], value = 'no', inline = True, label = 'Save samples:')
    saveModels = mo.ui.radio(options = ['no', 'yes'], value = 'no', inline = True, label = 'Save models:')
    saveFreqFootnote = mo.md('<div style="text-align: right">¹ <sub>Visualizations are always saved</sub></div>')
    # Work with existing model
    modelFileLabel = mo.md('Model:')
    modelFile = mo.ui.file(label = 'Upload', filetypes = ['.pt.zst'])
    createData = mo.ui.radio(options = ['yes', 'no'], value = 'yes', inline = True, label = 'Create data ²:')
    createDataFootnote = mo.md('<div style="text-align: right">² <sub>If no, continue training</sub></div>')

    # Advanced tab
    batchSize = mo.ui.number(start = 1, step = 1, value = params['batchSize'], label = 'Batch size:')
    lrGen = mo.ui.text(label = 'Generator learning rate:', value = str(params['lrGen']))
    lrDis = mo.ui.text(label = 'Discriminator learning rate:', value = str(params['lrDis']))
    loopCountGen = mo.ui.number(start = 1, step = 1, value = params['loopCountGen'], label = 'Generator loop count:')

    # Start
    getState, setState = mo.state(False)
    button = mo.ui.button(on_change = lambda _: setState(True), label = 'Start')
    return (
        appTitle,
        batchSize,
        button,
        createData,
        createDataFootnote,
        epochCount,
        getState,
        inputFile,
        inputFileLabel,
        loopCountGen,
        lrDis,
        lrGen,
        modelFile,
        modelFileLabel,
        outputFormat,
        projectName,
        saveFreq,
        saveFreqFootnote,
        saveModels,
        saveSamples,
        setState,
        useWandb,
    )


@app.cell
def _(
    batchSize,
    createData,
    createDataFootnote,
    epochCount,
    inputFile,
    inputFileLabel,
    loopCountGen,
    lrDis,
    lrGen,
    mo,
    modelFile,
    modelFileLabel,
    outputFormat,
    projectName,
    saveFreq,
    saveFreqFootnote,
    saveModels,
    saveSamples,
    useWandb,
):
    basicTab = mo.vstack([
        projectName,
        mo.hstack([
            inputFileLabel,
            inputFile,
        ], justify = 'start'),
        outputFormat,
        useWandb,
        epochCount,
        saveFreq,
        saveSamples,
        mo.hstack([
            saveModels,
            saveFreqFootnote
        ]),
        mo.md('---'),
        mo.accordion({
            '**Work with existing model**': mo.vstack([
                mo.hstack([
                    modelFileLabel,
                    modelFile
                ], justify = 'start'),
                mo.hstack([
                    createData,
                    createDataFootnote
                ])
            ])
        })
    ])

    advancedTab = mo.vstack([
        batchSize,
        lrGen,
        lrDis,
        loopCountGen,
        mo.md('---')
    ])

    tabs = mo.ui.tabs({
        '🔧 Basic': basicTab,
        '🛠️ Advanced': advancedTab
    })
    return advancedTab, basicTab, tabs


@app.cell
def _(appTitle, button, mo, tabs):
    mo.vstack([
        appTitle,
        tabs,
        button
    ])
    return


@app.cell
def _(
    batchSize,
    createData,
    epochCount,
    getState,
    inputFile,
    loopCountGen,
    lrDis,
    lrGen,
    mo,
    outputFormat,
    params,
    projectName,
    read_csv,
    run,
    saveFreq,
    saveModels,
    saveSamples,
    setState,
    to_boolean,
    useWandb,
):
    if getState():
        with mo.redirect_stdout():
            print('Processing...')
            params['outputFormat'] = outputFormat.value
            params['epochCount'] = epochCount.value
            params['saveFreq'] = saveFreq.value
            params['saveSamples'] = to_boolean(saveSamples.value)
            params['saveModels'] = to_boolean(saveModels.value)
            params['batchSize'] = batchSize.value
            params['lrGen'] = float(lrGen.value)
            params['lrDis'] = float(lrDis.value)
            params['loopCountGen'] = loopCountGen.value
            inputFileProcd = read_csv(inputFile)
            run(params, projectName.value, inputFileProcd, to_boolean(useWandb.value), None, to_boolean(createData.value), True)
            print('Done!')
            setState(False)
    return (inputFileProcd,)


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
