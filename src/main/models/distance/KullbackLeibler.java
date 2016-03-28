package main.models.distance;

import main.models.ClassifierModel;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Created by loongkuan on 26/03/16.
 **/
public class KullbackLeibler extends Distance{

    @Override
    public double findPvDistance(ClassifierModel model1, ClassifierModel model2, Instances allInstances) {
        double totalDist = 0.0f;
        for (Instance combination : allInstances){
            double Pi = model1.findPv(combination.toDoubleArray());
            double Qi = model2.findPv(combination.toDoubleArray());
            totalDist += (Pi * (Math.log(Pi / Qi) / Math.log(2)));
        }
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
                double Pi = model1.findPygv(classIndex, inst);
                double Qi = model2.findPygv(classIndex, inst);
                driftDist += (Pi * (Math.log(Pi / Qi) / Math.log(2)));
            }
            totalDist += driftDist;
        }
        totalDist = totalDist / allInstances.size();
        return totalDist;
    }
}
