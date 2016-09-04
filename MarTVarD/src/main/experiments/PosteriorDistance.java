package main.experiments;

import main.models.NaiveMatrix;
import weka.core.Instances;

import java.util.ArrayList;

/**
 * Created by LoongKuan on 31/07/2016.
 **/
public class PosteriorDistance extends Experiment{

    public PosteriorDistance(Instances instances1, Instances instances2, int nAttributesActive){
        super(instances1, instances2, nAttributesActive);
    }

    @Override
    public ArrayList<ExperimentResult> getResults(NaiveMatrix model1, NaiveMatrix model2, Instances allInstances) {
        double[] p = new double[allInstances.size()];
        double[] q = new double[allInstances.size()];
        double[] weights = new double[allInstances.size()];
        double[] separateDistance = new double[allInstances.size()];
        double[][] instanceValues = new double[allInstances.size()][allInstances.numAttributes()];
        for (int i = 0; i < allInstances.size(); i++) {
            weights[i] = (model1.findPv(allInstances.get(i)) + model2.findPv(allInstances.get(i))) / 2;
            p[i] = model1.findPyGv(allInstances.get(i));
            q[i] = model2.findPyGv(allInstances.get(i));
            separateDistance[i] = this.distanceMetric.findDistance(new double[]{p[i]}, new double[]{q[i]}, new double[]{weights[i]});
            instanceValues[i] = allInstances.get(i).toDoubleArray();
        }
        double finalDistance = this.distanceMetric.findDistance(p, q, weights);
        ExperimentResult finalResult = new ExperimentResult(finalDistance, separateDistance, instanceValues);
        ArrayList<ExperimentResult> returnResults = new ArrayList<>();
        returnResults.add(finalResult);
        return returnResults;
    }
}
