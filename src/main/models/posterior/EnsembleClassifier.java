package main.models.posterior;

import com.yahoo.labs.samoa.instances.WekaToSamoaInstanceConverter;
import main.models.distance.Distance;
import main.models.distance.HellingerDistance;
import moa.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.HashMap;
import java.util.stream.DoubleStream;

/**
 * Created by loongkuan on 1/04/16.
**/
public class EnsembleClassifier extends PosteriorModel{
    private Classifier currentClassifier;
    private Classifier[] baseClassifiers;
    private double[] switchPoints;
    private HashMap<Double, Integer> classValueToIndex = new HashMap<>();
    private WekaToSamoaInstanceConverter wekaConverter = new WekaToSamoaInstanceConverter();

    public EnsembleClassifier(EnsembleClassifier baseModel, Instances dataSet) {
        this.dataSet = dataSet;
        this.baseClassifiers = new Classifier[baseModel.baseClassifiers.length];
        for (int i = 0; i < baseModel.baseClassifiers.length; i++) {
            this.baseClassifiers[i] = baseModel.baseClassifiers[i].copy();
        }
        this.switchPoints = baseModel.switchPoints.clone();
        reset();
    }

    public EnsembleClassifier(Classifier[] baseClassifiers, double[] switchPoints, Instances dataSet) {
        this.dataSet = dataSet;
        this.baseClassifiers = baseClassifiers;
        this.switchPoints = switchPoints;
    }

    private void getClassLabels() {
        for (Instance inst : this.dataSet) {
            classValueToIndex.put(inst.classValue(), inst.classIndex());
        }
    }

    @Override
    public void setData(Instances data) {
        this.dataSet = data;
        this.reset();
    }

    @Override
    public EnsembleClassifier copy() {
        Classifier[] classifier_copies = new Classifier[baseClassifiers.length];
        for (int i = 0; i < baseClassifiers.length; i++) {
            classifier_copies[i].resetLearning();
            classifier_copies[i] = baseClassifiers[i].copy();
        }
        reset();
        return new EnsembleClassifier(classifier_copies, switchPoints.clone(), dataSet);
    }

    @Override
    public void reset() {
        for (Classifier baseClassifier : this.baseClassifiers) {
            baseClassifier.resetLearning();
            baseClassifier.prepareForUse();
            for (Instance inst : dataSet) {
                baseClassifier.trainOnInstance(wekaConverter.samoaInstance(inst));
            }
        }
        getClassLabels();
        currentClassifier = baseClassifiers[0];
    }

    @Override
    public double findPyGv(double classValue, Instance vector) {
        // Check if class given is in data set
        if (!this.classValueToIndex.containsKey(classValue)) return 0.0f;
        else {
            // Get classifier's votes for instance/vector
            double[] votesForInstance = currentClassifier.getVotesForInstance(wekaConverter.samoaInstance(vector));
            // Normalise votes
            double sumVotes = DoubleStream.of(votesForInstance).sum();
            double[] normVotes = new double[votesForInstance.length];
            for (int j = 0; j < votesForInstance.length; j++) {
                normVotes[j] = (sumVotes == 0.0) ? 0.0f : votesForInstance[j] / sumVotes;
            }
            return normVotes[this.classValueToIndex.get(classValue)];
        }
    }

    @Override
    public double findDistance(PosteriorModel model1, PosteriorModel model2, Instances domain) {
        Double[] classValues = this.classValueToIndex.keySet().toArray(new Double[classValueToIndex.size()]);
        Distance distanceMetric = new HellingerDistance();
        double[] driftMags = new double[this.baseClassifiers.length];

        for (Instance inst : domain) {
            double[] p = new double[classValues.length];
            double[] q = new double[classValues.length];
            for (int j = 0; j < baseClassifiers.length; j++) {
                this.currentClassifier = this.baseClassifiers[j];
                for (int k = 0; k < classValues.length; k++) {
                    p[k] = model1.findPyGv(classValues[k], inst);
                    q[k] = model2.findPyGv(classValues[k], inst);
                }
                driftMags[j] += distanceMetric.findDistance(p, q);
            }
        }

        for (int i = 0; i < this.baseClassifiers.length; i++) {
            driftMags[i] = driftMags[i]/domain.size();
            if (driftMags[i] <= this.switchPoints[i]) return driftMags[i];
        }
        System.out.println("Fatal Error Occurred");
        return 0.0f;
    }
}
