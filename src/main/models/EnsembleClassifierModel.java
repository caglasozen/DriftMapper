package main.models;

import moa.classifiers.Classifier;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.stream.DoubleStream;

/**
 * Created by Lee on 30/01/2016.
 **/
public class EnsembleClassifierModel extends ClassifierModel{
    protected Classifier[] baseClassifiers;
    // The p(y|v) when to switch to the next classifier
    protected double[] switchPoints;

    @Override
    public EnsembleClassifierModel copy() {
        Classifier[] classifier_copies = new Classifier[baseClassifiers.length];
        for (int i = 0; i < baseClassifiers.length; i++) {
            classifier_copies[i] = baseClassifiers[i].copy();
        }
        return new EnsembleClassifierModel(classifier_copies, switchPoints.clone(), dataSet);
    }

    public EnsembleClassifierModel(EnsembleClassifierModel baseModel, Instances dataSet) {
        this.dataSet = dataSet;
        this.baseClassifiers = new Classifier[baseModel.baseClassifiers.length];
        for (int i = 0; i < baseModel.baseClassifiers.length; i++) {
            this.baseClassifiers[i] = baseModel.baseClassifiers[i].copy();
        }
        this.switchPoints = baseModel.switchPoints.clone();
        getXMeta();
        resetModel();
    }

    public EnsembleClassifierModel(Classifier[] baseClassifiers, double[] switchPoints, Instances dataSet) {
        this.dataSet = dataSet;
        this.baseClassifiers = baseClassifiers;
        this.switchPoints = switchPoints;
        if (dataSet.numAttributes() != 0) {
            getXMeta();
            resetModel();
        }
    }

    public void changeClassifiers(Classifier[] classifiers, double[] switchPoints) {
        this.baseClassifiers = classifiers;
        this.switchPoints = switchPoints;
        resetModel();
    }

    protected void trainClassifier() {
        for (Classifier baseClassifier : this.baseClassifiers) {
            baseClassifier.resetLearning();
            baseClassifier.prepareForUse();
            for (int i = 0; i < dataSet.size(); i++) {
                baseClassifier.trainOnInstance(wekaConverter.samoaInstance(dataSet.get(i)));
            }
        }
    }

    @Override
    public double findPygv(int classLabelIndex, Instance inst) {
        double pygv = 0.0f;
        for (int i = 0; i < this.baseClassifiers.length; i++) {
            // Get classifier's votes for instance/vector
            double[] votesForInstance = baseClassifiers[i].getVotesForInstance(wekaConverter.samoaInstance(inst));
            // Normalise votes
            double sumVotes = DoubleStream.of(votesForInstance).sum();
            double[] normVotes = new double[votesForInstance.length];
            for (int j = 0; j < votesForInstance.length; j++) {
                normVotes[j] = (sumVotes == 0.0) ? 0.0f : votesForInstance[j] / sumVotes;
            }

            pygv = normVotes[classLabelIndex];

            if (pygv < this.switchPoints[i]) {
                return pygv;
            }
        }
        return pygv;
    }

    /**
     * Calculate the probability of a certain combination of x values (aka. xVector) occurring
     * @param xValVector The combination of x values (xVector)
     * @return Probability of given vector of x values occurring
     */
    @Override
    public double findPv(double[] xValVector) {
        try {
            int[] dataVector = new int[xValVector.length];
            for (int i = 0; i < xValVector.length; i++) {
                dataVector[i] = this.bnModel.nodes[i].getOutcomeIndex(Double.toString(xValVector[i]));
                //dataVector[i] = Double.valueOf(xValVector[i]).intValue();
            }
            return bnModel.getJointProbabilityOfX(dataVector);
        }
        catch (IllegalArgumentException ex) {
            System.out.print("e ");
            return 0.0f;
        }
    }
}
