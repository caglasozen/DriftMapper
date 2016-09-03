package main.experiments;

import main.distance.Distance;
import main.distance.TotalVariation;
import main.models.NaiveMatrix;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Created by LoongKuan on 31/07/2016.
 **/
public class CovariateDistance extends Experiment{

    public CovariateDistance(Instances instances1, Instances instances2, int nAttributesActive){
        super(instances1, instances2, nAttributesActive);
    }

    @Override
    public ExperimentResult getResults(NaiveMatrix model1, NaiveMatrix model2, Instances allInstances) {
        double[] p = new double[allInstances.size()];
        double[] q = new double[allInstances.size()];
        double[] separateDistance = new double[allInstances.size()];
        double[][] instanceValues = new double[allInstances.size()][allInstances.numAttributes()];
        for (int i = 0; i < allInstances.size(); i++) {
            p[i] = model1.findPv(allInstances.get(i));
            q[i] = model2.findPv(allInstances.get(i));
            separateDistance[i] = this.distanceMetric.findDistance(new double[]{p[i]}, new double[]{q[i]});
            instanceValues[i] = allInstances.get(i).toDoubleArray();
            instanceValues[i][allInstances.classIndex()] = -1.0f;
        }
        double finalDistance = this.distanceMetric.findDistance(p, q);
        return(new ExperimentResult(finalDistance, separateDistance, instanceValues));
    }
}
