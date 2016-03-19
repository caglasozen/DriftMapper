package main;

import main.generator.CategoricalDriftGenerator;
import main.models.EnsembleClassifierModel;
import com.yahoo.labs.samoa.instances.SamoaToWekaInstanceConverter;
import weka.core.Instances;

/**
 * Created by loongkuan on 11/03/16.
 **/
public class Experiments {
    EnsembleClassifierModel[] models;
    Instances allInstances;

    public Experiments(EnsembleClassifierModel baseModel, Instances allInstances, int windowSize, boolean startEnd) {
        System.out.println("Initialising experiment...");
        if (!startEnd) models = new EnsembleClassifierModel[allInstances.size() / windowSize];
        else models = new EnsembleClassifierModel[2];
        this.allInstances = allInstances;
        trainAllModels(baseModel, windowSize);
    }

    public Experiments(EnsembleClassifierModel baseModel, CategoricalDriftGenerator dataStream) {
        System.out.println("Initialising experiment...");
        models = new EnsembleClassifierModel[2];
        this.allInstances = convertStreamToInstances(dataStream);
        trainAllModels(baseModel, dataStream.burnInNInstances.getValue());
    }

    private void trainAllModels(EnsembleClassifierModel baseModel, int windowSize) {
        System.out.println("Training Models...");
        for (int i = 0; i < models.length; i++) {
            models[i] = new EnsembleClassifierModel(baseModel,
                    new Instances(this.allInstances, i*windowSize, windowSize));
        }
    }

    private static Instances convertStreamToInstances(CategoricalDriftGenerator dataStream) {
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
        return new String[]{Double.toString(EnsembleClassifierModel.pvModelDistance(models[0], models[models.length - 1])),
                Double.toString(EnsembleClassifierModel.pygvModelDistance(models[0], models[models.length - 1]))};
    }

    public String[][] distanceOverInstances() {
        String[][] distances = new String[models.length][2];
        for (int i = 0; i < models.length; i++) {
            distances[i][0] = Double.toString(EnsembleClassifierModel.pvModelDistance(models[0], models[i]));
            distances[i][1] = Double.toString(EnsembleClassifierModel.pygvModelDistance(models[0], models[i]));
        }
        return distances;
    }
}
