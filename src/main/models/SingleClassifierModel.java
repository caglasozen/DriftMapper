package main.models;

import moa.classifiers.Classifier;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.stream.DoubleStream;

/**
 * Created by Lee on 30/01/2016.
 **/
public class SingleClassifierModel extends ClassifierModel{
    protected Classifier baseClassifier;

    public SingleClassifierModel copy() {
        return new SingleClassifierModel(baseClassifier.copy(), dataSet);
    }

    public SingleClassifierModel(Classifier baseClassifier, Instances dataSet) {
        this.dataSet = dataSet;
        this.baseClassifier = baseClassifier;
        getXMeta();
        resetModel();
    }

    public void changeClassifier(Classifier classifier) {
        baseClassifier = classifier;
        resetModel();
    }

    protected void trainClassifier() {
        baseClassifier.resetLearning();
        baseClassifier.prepareForUse();
        for (int i = 0; i < dataSet.size(); i++) {
            baseClassifier.trainOnInstance(wekaConverter.samoaInstance(dataSet.get(i)));
        }
    }

    @Override
    public double findPygv(String classLabel, String[] xValVector) {
        double pygv = 0.0f;

        // Convert Array of vector given t an instance
        Instance inst = new DenseInstance(xValVector.length);
        for (int i = 0; i < xValVector.length; i++) {
            inst.setValue(i, Double.parseDouble(xValVector[i]));
        }

        // Get classifier's votes for instance/vector
        double[] votesForInstance = baseClassifier.getVotesForInstance(wekaConverter.samoaInstance(inst));
        // Normalise votes
        double sumVotes = DoubleStream.of(votesForInstance).sum();
        double[] normVotes = new double[votesForInstance.length];
        for (int i = 0; i < votesForInstance.length; i++) {
            normVotes[i] = (sumVotes == 0.0) ? 0.0f : votesForInstance[i] / sumVotes;
        }

        pygv = normVotes[(int)Double.parseDouble(classLabel)];

        return pygv;
    }

    /**
     * Calculate the probability of a certain combination of x values (aka. xVector) occurring
     * @param xValVector The combination of x values (xVector)
     * @return Probability of given vector of x values occurring
     */
    @Override
    public double findPv(double[] xValVector) {
        int[] dataVector = new int[xValVector.length];
        for (int i = 0; i < xValVector.length; i++) {
            dataVector[i] = Double.valueOf(xValVector[i]).intValue();
        }
        return bnModel.getJointProbabilityOfX(dataVector);
    }
}
