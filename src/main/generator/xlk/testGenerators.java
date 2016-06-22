package main.generator.xlk;

import com.opencsv.CSVWriter;
import com.yahoo.labs.samoa.instances.SamoaToWekaInstanceConverter;
import jdk.nashorn.internal.scripts.JO;
import main.Experiments;
import main.generator.AbruptTreeDriftGenerator;
import main.generator.CategoricalDriftGenerator;
import main.models.JointModel;
import main.models.posterior.PosteriorModel;
import main.models.posterior.SingleClassifier;
import main.models.prior.BayesianNetwork;
import main.models.prior.PriorModel;
import main.models.sampling.AbstractSampler;
import main.models.sampling.RandomSamples;
import moa.classifiers.meta.WEKAClassifier;
import moa.classifiers.trees.HoeffdingTree;
import moa.streams.InstanceStream;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.core.Instance;
import weka.core.Instances;
import weka.gui.beans.Join;

import java.io.FileWriter;
import java.io.IOException;

/**
 * Created by loongkuan on 31/05/16.
 */
public class testGenerators {
    public static void main(String[] args){
        WEKAClassifier ensembleClassifier = new WEKAClassifier();
        ensembleClassifier.baseLearnerOption.setCurrentObject(new RandomTree());
        com.yahoo.labs.samoa.instances.SamoaToWekaInstanceConverter conv = new com.yahoo.labs.samoa.instances.SamoaToWekaInstanceConverter();
        AbruptTreeDriftGenerator gen = new AbruptTreeDriftGenerator();
        gen.prepareForUse();
        Instance inst = conv.wekaInstance(gen.nextInstance().instance);
        Instances dataset = new Instances(inst.dataset());
        dataset.add(conv.wekaInstance(gen.nextInstance().instance));
        PosteriorModel basePosterior = new SingleClassifier(new HoeffdingTree(),dataset);
        PriorModel basePrior = new BayesianNetwork(dataset);
        AbstractSampler sampler = new RandomSamples(dataset, 1000, 0L);
        //AbstractSampler sampler = new AllSamples(dataset);
        JointModel baseModel = new JointModel(basePrior, basePosterior, sampler);

        classifierDistanceTest(baseModel);
    }

    public static void writeToCSV(String[][] data, String[] header, String filename) throws IOException {
        CSVWriter writer = new CSVWriter(new FileWriter(filename), ',');
        // feed in your array (or convert your data to an array)
        writer.writeNext(header);
        for (String[] dataLine : data) {
            writer.writeNext(dataLine);
        }
        writer.close();
    }

    public static Instances convertStreamToInstances(AbruptDriftGenerator dataStream) {
        SamoaToWekaInstanceConverter converter = new SamoaToWekaInstanceConverter();
        Instances convertedStreams = new Instances(converter.wekaInstancesInformation(dataStream.getHeader()),
                dataStream.burnInNInstances.getValue());
        for (int i = 0; i < dataStream.burnInNInstances.getValue()*2; i++) {
            convertedStreams.add(converter.wekaInstance(dataStream.nextInstance().getData()));
        }
        return convertedStreams;
    }

    public static Instances convertStreamToInstances(IncrementalDriftGenerator dataStream) {
        SamoaToWekaInstanceConverter converter = new SamoaToWekaInstanceConverter();
        Instances convertedStreams = new Instances(converter.wekaInstancesInformation(dataStream.getHeader()),
                dataStream.burnInNInstances.getValue());
        for (int i = 0; i < dataStream.burnInNInstances.getValue()*2 + 1000; i++) {
            convertedStreams.add(converter.wekaInstance(dataStream.nextInstance().getData()));
        }
        return convertedStreams;
    }

    private static void classifierDistanceTest(JointModel baseModel) {
        AbruptDriftGenerator generator1 = new AbruptDriftGenerator();
        generator1.driftConditional.setValue(true);
        generator1.driftPriors.setValue(true);
        IncrementalDriftGenerator generator2 = new IncrementalDriftGenerator();
        generator2.driftConditional.setValue(true);
        generator2.driftPriors.setValue(true);
        double[] driftMags = new double[]{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
        int[] dataPoints = new int []{100000, 1000000};
        int nTests = 10;

        JointModel model1 = baseModel.copy();
        JointModel model2 = baseModel.copy();

        for (int n = 0; n < 2; n++) {
            String[][] pvResults = new String[driftMags.length][dataPoints.length];
        String[][] pyResults = new String[driftMags.length][dataPoints.length];

        for (int i = 0; i < driftMags.length; i++) {
            double dm = driftMags[i];
            if (n==0) generator1.driftMagnitude.setValue(dm);
            else generator2.driftMagnitude.setValue(dm);

            for (int j = 0; j < dataPoints.length; j++) {
                int nData = dataPoints[j];
                if (n==0) generator1.burnInNInstances.setValue(nData);
                else generator2.burnInNInstances.setValue(nData);
                //System.out.println("Drift Magnitude: " + dm);
                //System.out.println("No. Instances Before/After Drift: " + nData);

                double avgPygv = 0.0f;
                double avgPv = 0.0f;
                for (int k = 0; k < nTests; k++) {
                    //System.out.println("Run: " + (k+1) + "\t");
                    Instances allInstances;
                    if (n == 0) {
                        generator1.restart();
                        generator1.prepareForUse();
                        allInstances = convertStreamToInstances(generator1);
                        model1.setData(new Instances(allInstances, 0, nData));
                        model2.setData(new Instances(allInstances, nData, nData));
                    }
                    else {
                        generator2.restart();
                        generator2.prepareForUse();
                        allInstances = convertStreamToInstances(generator2);
                        model1.setData(new Instances(allInstances, 0, nData));
                        model2.setData(new Instances(allInstances, nData + 1000, nData));
                    }
                    avgPv += JointModel.pvModelDistance(model1, model2);
                    avgPygv += JointModel.pyGvModelDistance(model1, model2);
                }
                avgPv /= nTests;
                avgPygv /= nTests;
                //System.out.println("p(y|X) drift = " + avgPygv);
                pvResults[i][j] = Double.toString(avgPv);
                pyResults[i][j] = Double.toString(avgPygv);
                System.out.println("Estimated p(X): " + avgPv);
                System.out.println("Estimated p(y|X): " + avgPygv);
            }
        }
        try {
            String name = (n==0) ? "Abrupt" : "Incremental";
            writeToCSV(pvResults, new String[]{"100000","1000000"}, name + "_pv.csv");
            writeToCSV(pyResults, new String[]{"100000","1000000"}, name + "_pygv.csv");
        }
        catch (Exception ex){
            ex.printStackTrace();
        }
        }

    }

}
