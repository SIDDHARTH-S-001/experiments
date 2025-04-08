import adsk.core, adsk.fusion, adsk.cam, traceback

# --- Import and run FusionDebugSetup ---
try:
    from fusionDebugSetup import FusionDebugSetup
    pre_run_script_path = r'C:\Users\SIDDHARTH\AppData\Roaming\Autodesk\Autodesk Fusion 360\API\Python\vscode\pre-run.py'
    debug_setup = FusionDebugSetup(pre_run_script_path)
    debug_setup.run()
except Exception as debug_setup_error:
    print(f"[Warning] Failed to execute FusionDebugSetup: {debug_setup_error}")

# --- Attach Debugger ---
try:
    import debugpy
    debugpy.listen(('localhost', 9000))
    debugpy.wait_for_client()  # Optional: pause until VS Code attaches
except:
    pass

# Global list to keep event handler references.
handlers = []

# ===============================================
# COMMAND DIALOG FOR CUBE CREATION WITH OPTIONS
# ===============================================
class CubeCreatorCommandCreatedHandler(adsk.core.CommandCreatedEventHandler):
    def notify(self, args):
        try:
            cmd = args.command
            inputs = cmd.commandInputs

            # --- Dropdown for units ---
            dropdownUnits = inputs.addDropDownCommandInput(
                'unitsInput',
                'Units',
                adsk.core.DropDownStyles.TextListDropDownStyle
            )
            dropdownUnits.listItems.add('mm', True)  # default selected
            dropdownUnits.listItems.add('cm', False)
            dropdownUnits.listItems.add('inch', False)
            dropdownUnits.listItems.add('m', False)

            # --- Text boxes for dimensions (L, W, H) ---
            inputs.addStringValueInput('lengthInput', 'Length', '10')
            inputs.addStringValueInput('widthInput', 'Width', '10')
            inputs.addStringValueInput('heightInput', 'Height', '10')

            # --- Dropdown for feature type (None, Fillet, Chamfer) ---
            dropdownFeature = inputs.addDropDownCommandInput(
                'featureType',
                'Feature Type',
                adsk.core.DropDownStyles.TextListDropDownStyle
            )
            dropdownFeature.listItems.add('None', True)   # default is None
            dropdownFeature.listItems.add('Fillet', False)
            dropdownFeature.listItems.add('Chamfer', False)

            # --- Checkbox for selection: Apply to Edges ---
            inputs.addBoolValueInput('edgeOption', 'Apply to Edges', True, '', True)

            # --- Text box for the feature value (radius/distance) ---
            # Initially hidden if 'None' is selected.
            featureValueInput = inputs.addStringValueInput('featureValue', 'Feature Value', '1')
            featureValueInput.isVisible = False

            # Add input changed handler to toggle featureValue visibility.
            onInputChanged = CubeCreatorCommandInputChangedHandler()
            cmd.inputChanged.add(onInputChanged)
            handlers.append(onInputChanged)

            # Add the execute handler.
            onExecute = CubeCreatorCommandExecuteHandler()
            cmd.execute.add(onExecute)
            handlers.append(onExecute)
        except Exception as e:
            app = adsk.core.Application.get()
            ui  = app.userInterface
            ui.messageBox('Failed in command created handler:\n{}'.format(traceback.format_exc()))

class CubeCreatorCommandInputChangedHandler(adsk.core.InputChangedEventHandler):
    def notify(self, args):
        try:
            eventArgs = adsk.core.InputChangedEventArgs.cast(args)
            changedInput = eventArgs.input
            if changedInput.id == 'featureType':
                cmdInputs = eventArgs.firingEvent.sender.commandInputs
                featureValueInput = cmdInputs.itemById('featureValue')
                # Toggle visibility: show only if Fillet or Chamfer is selected.
                if changedInput.selectedItem.name == 'None':
                    featureValueInput.isVisible = False
                else:
                    featureValueInput.isVisible = True
        except Exception as e:
            app = adsk.core.Application.get()
            ui = app.userInterface
            ui.messageBox('Failed in input changed handler:\n{}'.format(traceback.format_exc()))

class CubeCreatorCommandExecuteHandler(adsk.core.CommandEventHandler):
    def notify(self, args):
        # NOTE: Do not catch exceptions here that we intend to propagate so that the command dialog remains open.
        app = adsk.core.Application.get()
        ui = app.userInterface
        design = app.activeProduct
        rootComp = design.rootComponent
        inputs = args.command.commandInputs

        unitItem = inputs.itemById('unitsInput').selectedItem.name
        length_val = float(inputs.itemById('lengthInput').value)
        width_val  = float(inputs.itemById('widthInput').value)
        height_val = float(inputs.itemById('heightInput').value)
        featureType = inputs.itemById('featureType').selectedItem.name
        edgeOption = inputs.itemById('edgeOption').value

        # Only retrieve feature value if needed.
        if featureType == 'None':
            feature_val = 0
        else:
            feature_val = float(inputs.itemById('featureValue').value)

        # Conversion factors (assumes Fusion's internal units are in centimeters).
        conversion_factors = {'mm': 0.1, 'cm': 1.0, 'inch': 2.54, 'm': 100.0}
        factor = conversion_factors.get(unitItem, 1.0)
        length_converted = length_val * factor
        width_converted  = width_val  * factor
        height_converted = height_val * factor
        feature_converted = feature_val * factor

        # Create a new sketch on the XY plane.
        sketches = rootComp.sketches
        xyPlane = rootComp.xYConstructionPlane
        sketch = sketches.add(xyPlane)

        # Draw a rectangle for the base of the cube.
        pt1 = adsk.core.Point3D.create(0, 0, 0)
        pt2 = adsk.core.Point3D.create(length_converted, width_converted, 0)
        sketch.sketchCurves.sketchLines.addTwoPointRectangle(pt1, pt2)

        # Extrude the rectangle to create a cube.
        prof = sketch.profiles.item(0)
        extrudes = rootComp.features.extrudeFeatures
        extInput = extrudes.createInput(prof, adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
        heightInputObj = adsk.core.ValueInput.createByReal(height_converted)
        extInput.setDistanceExtent(False, heightInputObj)
        extFeature = extrudes.add(extInput)
        cubeBody = extFeature.bodies.item(0)

        # Apply feature if chosen (and if edgeOption is true).
        if featureType != 'None' and edgeOption:
            try:
                if featureType == 'Fillet':
                    filletFeats = rootComp.features.filletFeatures
                    filletInput = filletFeats.createInput()

                    edgesCollection = adsk.core.ObjectCollection.create()
                    for edge in cubeBody.edges:
                        edgesCollection.add(edge)
                    filletInput.addConstantRadiusEdgeSet(
                        edgesCollection,
                        adsk.core.ValueInput.createByReal(feature_converted),
                        True
                    )
                    filletFeats.add(filletInput)
                elif featureType == 'Chamfer':
                    chamferFeats = rootComp.features.chamferFeatures
                    chamferInput = chamferFeats.createInput2()

                    edgesCollection = adsk.core.ObjectCollection.create()
                    for edge in cubeBody.edges:
                        edgesCollection.add(edge)

                    # For older API versions, assign edges then set equal distance.
                    chamferInput.edges = edgesCollection
                    chamferInput.setToEqualDistance(adsk.core.ValueInput.createByReal(feature_converted))
                    chamferFeats.add(chamferInput)
            except Exception as featureErr:
                ui.messageBox("Observed conflict with provided values, hence generating a simple cube, try again considering the following\nSolution:\n1) Check the dimensions and feature value provided\n2) Else choose \"None\" features and generate a simple cube")
                design.undo()  # Roll back cube creation.
                # Do not terminate the plugin; allow the user to adjust values.
                return

        # If we reached this point, the cube (with feature if any) was generated successfully.
        adsk.terminate()

# ===============================================
# ENTRY POINT: CREATE AND EXECUTE THE COMMAND
# ===============================================
def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui  = app.userInterface

        # Create (or get) a command definition.
        cmdDef = ui.commandDefinitions.itemById('CubeCreatorCommand')
        if not cmdDef:
            cmdDef = ui.commandDefinitions.addButtonDefinition(
                'CubeCreatorCommand',
                'Cube Creator',
                'Creates a cube with fillet/chamfer options'
            )

        onCommandCreated = CubeCreatorCommandCreatedHandler()
        cmdDef.commandCreated.add(onCommandCreated)
        handlers.append(onCommandCreated)

        # Execute the command.
        cmdDef.execute()

        # We set adsk.autoTerminate(False) so the script doesn't end before the command completes.
        adsk.autoTerminate(False)
    except Exception as e:
        if ui:
            ui.messageBox('Failed in run:\n{}'.format(traceback.format_exc()))
