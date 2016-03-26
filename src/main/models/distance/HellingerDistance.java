package main.models.distance;

import main.models.ClassifierModel;
import moa.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Created by loongkuan on 26/03/16.
 **/
public class HellingerDistance extends Distance{

    @Override
    public double findPvDistance(ClassifierModel model1, ClassifierModel model2, Instances allInstances) {
        double totalDist = 0.0f;
        for (Instance combination : allInstances){
            totalDist = totalDist +
                    Math.pow((Math.sqrt(model1.findPv(combination.toDoubleArray()))
                            - Math.sqrt(model2.findPv(combination.toDoubleArray()))), 2);
        }
        totalDist = Math.sqrt(totalDist);
        totalDist = totalDist * (1/Math.sqrt(2));
        return totalDist;
    }

    @Override
    public double findPyGvDistance(ClassifierModel model1, ClassifierModel model2, Instances allInstances) {
        double totalDist = 0.0f;
        for (Instance inst : allInstances) {
            int numClasses = (model1.dataSet.numClasses()< model2.dataSet.numClasses()) ?
                    model2.dataSet.numClasses() : model1.dataSet.numClasses();
            double driftDist = 0.0f;
            for (int classIndex = 0; classIndex < numClasses; classIndex++) {
                driftDist = driftDist +
                        Math.pow((Math.sqrt(model1.findPygv(classIndex, inst)) -
                                Math.sqrt(model2.findPygv(classIndex, inst))), 2);
            }
            driftDist = Math.sqrt(driftDist);
            driftDist = driftDist * (1/Math.sqrt(2));
            totalDist = totalDist + driftDist;
        }

        totalDist = totalDist / allInstances.size();
        return totalDist;
    }
}
