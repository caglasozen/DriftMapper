package main;

import main.generator.CategoricalDriftGenerator;
import main.models.JointModel;
import com.yahoo.labs.samoa.instances.SamoaToWekaInstanceConverter;
import weka.core.Instances;

/**
 * Created by loongkuan on 11/03/16.
 **/
//TODO: Use something nicer to manage analyse
public class Experiments {
    private JointModel[] models;
    private Instances allInstances;

    public Experiments(JointModel baseModel, Instances allInstances, int windowSize, boolean startEnd) {
        System.out.println("Initialising experiment...");
        if (!startEnd) models = new JointModel[allInstances.size() / windowSize];
        else models = new JointModel[2];
        this.allInstances = allInstances;
        trainAllModels(baseModel, windowSize);
    }

    public Experiments(JointModel baseModel, CategoricalDriftGenerator dataStream) {
        System.out.println("Initialising experiment...");
        models = new JointModel[2];
        this.allInstances = convertStreamToInstances(dataStream);
        trainAllModels(baseModel, dataStream.burnInNInstances.getValue());
    }

    private void trainAllModels(JointModel baseModel, int windowSize) {
        System.out.println("Training Models...");
        for (int i = 0; i < models.length; i++) {
            models[i] = new JointModel(baseModel);
            models[i].setData(new Instances(this.allInstances, i*windowSize, windowSize));
        }
    }

    public static Instances convertStreamToInstances(CategoricalDriftGenerator dataStream) {
        SamoaToWekaInstanceConverter converter = new SamoaToWekaInstanceConverter();
        Instances convertedStreams = new Instances(converter.wekaInstancesInformation(dataStream.getHeader()),
                dataStream.burnInNInstances.getValue());
        for (int i = 0; i < dataStream.burnInNInstances.getValue()*2; i++) {
            convertedStreams.add(converter.wekaInstance(dataStream.nextInstance().getData()));
        }
        return convertedStreams;
    }

    public String[] distanceBetweenStartEnd() {
        System.out.println("Running Experiment...");
        return new String[]{Double.toString(JointModel.pvModelDistance(models[0], models[models.length - 1])),
                Double.toString(JointModel.pyGvModelDistance(models[0], models[models.length - 1]))};
    }

    public String[][] distanceToStartOverInstances() {
        String[][] distances = new String[models.length][2];
        System.out.println("Number of models: " + models.length);
        for (int i = 0; i < models.length; i++) {
            distances[i][0] = Double.toString(JointModel.pvModelDistance(models[0], models[i]));
            distances[i][1] = Double.toString(JointModel.pyGvModelDistance(models[0], models[i]));
        }
        return distances;
    }

    public String[][] distanceToPrevOverInstances() {
        String[][] distances = new String[models.length - 1][2];
        System.out.println("Number of models: " + models.length);
        for (int i = 0; i < models.length - 1; i++) {
            distances[i][0] = Double.toString(JointModel.pvModelDistance(models[i], models[i + 1]));
            distances[i][1] = Double.toString(JointModel.pyGvModelDistance(models[i], models[i + 1]));
        }
        return distances;
    }
}
