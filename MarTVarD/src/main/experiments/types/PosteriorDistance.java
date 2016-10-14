package main.experiments.types;

import main.experiments.Experiment;
import main.experiments.ExperimentResult;
import main.models.NaiveMatrix;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;

/**
 * Created by LoongKuan on 31/07/2016.
 **/
public class PosteriorDistance extends Experiment {

    public PosteriorDistance(Instances instances1, Instances instances2, int nAttributesActive, int[] attributeIndices, int sampleSize, int nTests){
        super(instances1, instances2, nAttributesActive, attributeIndices, new int[]{instances1.classIndex()}, sampleSize, nTests);
    }

    @Override
    public ArrayList<ExperimentResult> getResults(NaiveMatrix model1, NaiveMatrix model2, Instances allInstances, double sampleScale) {
        double[] p = new double[allInstances.size()];
        double[] q = new double[allInstances.size()];
        double[] weights = new double[allInstances.size()];
        double[] separateDistance = new double[allInstances.size()];
        double[][] instanceValues = new double[allInstances.size()][allInstances.numAttributes()];
        for (int i = 0; i < allInstances.size(); i++) {
            Instance covariateInstance = new DenseInstance(allInstances.get(i));
            covariateInstance.setDataset(allInstances);
            covariateInstance.setMissing(covariateInstance.classIndex());
            weights[i] = (model1.findPv(covariateInstance) + model2.findPv(covariateInstance)) / 2;
            p[i] = model1.findPyGv(allInstances.get(i));
            q[i] = model2.findPyGv(allInstances.get(i));
            separateDistance[i] = this.distanceMetric.findDistance(new double[]{p[i]}, new double[]{q[i]}, new double[]{weights[i]});
            instanceValues[i] = allInstances.get(i).toDoubleArray();
        }
        double finalDistance = this.distanceMetric.findDistance(p, q, weights) * sampleScale;
        ExperimentResult finalResult = new ExperimentResult(finalDistance, separateDistance, instanceValues);
        ArrayList<ExperimentResult> returnResults = new ArrayList<>();
        returnResults.add(finalResult);
        return returnResults;
    }

    @Override
    public String[][] getResultTable() {
        return this.getResultTable(0, "*");
    }
}
