package main.experiments;

import main.models.NaiveMatrix;
import weka.core.Instances;

/**
 * Created by LoongKuan on 31/07/2016.
 **/
public class ConditionedCovariateDistance extends Experiment{

    public ConditionedCovariateDistance(Instances instances1, Instances instances2, int nAttributesActive){
        super(instances1, instances2, nAttributesActive);
    }

    @Override
    public ExperimentResult getResults(NaiveMatrix model1, NaiveMatrix model2, Instances allInstances) {
        double[] p = new double[allInstances.size()];
        double[] q = new double[allInstances.size()];
        double[] weights = new double[allInstances.size()];
        double[] separateDistance = new double[allInstances.size()];
        double[][] instanceValues = new double[allInstances.size()][allInstances.numAttributes()];
        for (int i = 0; i < allInstances.size(); i++) {
            p[i] = model1.findPvGy(allInstances.get(i));
            q[i] = model2.findPvGy(allInstances.get(i));
            weights[i] = (model1.findPy(allInstances.get(i)) + model2.findPy(allInstances.get(i))) / 2;
            separateDistance[i] = this.distanceMetric.findDistance(new double[]{p[i]}, new double[]{q[i]}, new double[]{weights[i]});
            instanceValues[i] = allInstances.get(i).toDoubleArray();
        }
        double finalDistance = this.distanceMetric.findDistance(p, q, weights);
        return(new ExperimentResult(finalDistance, separateDistance, instanceValues));
    }
}
