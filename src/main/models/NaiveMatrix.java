package main.models;

import main.models.distance.Distance;
import main.models.distance.TotalVariation;
import main.models.posterior.PosteriorModel;
import main.models.prior.PriorModel;
import main.models.sampling.AbstractSampler;
import main.models.sampling.AllSamples;
import org.apache.commons.lang3.ArrayUtils;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * Created by loongkuan on 11/05/16.
 **/
public class NaiveMatrix extends AbstractModel implements PriorModel, PosteriorModel{
    private int[][] combinationFreq;

    private double[][] max;
    private double[][] min;
    private double[] mean;
    private double[] sd;

    public NaiveMatrix(Instances dataSet) {
        this.setDataSet(dataSet);
        this.reset();
    }

    public NaiveMatrix(AbstractSampler sampler) {
        this.setSampler(sampler);
        this.reset();
    }

    @Override
    public void reset() {
        int nClasses = this.sampler.getAllClasses().size();

        int nCombinations = 1;
        ArrayList<ArrayList<String>> attributesValues = this.sampler.getAllPossibleValues();
        for (ArrayList<String> attributeValue : attributesValues) {
            nCombinations *= attributeValue.size();
        }
        combinationFreq = new int[nCombinations][nClasses + 1];
        for (Instance inst : this.sampler.getDataSet()) {
            double[] instValues = Arrays.copyOfRange(inst.toDoubleArray(), 0, inst.numAttributes() - 1);
            int hash = this.convertInstToHash(instValues);
            int classHash = this.convertClassToHash(inst.toDoubleArray()[inst.numAttributes() - 1]);
            this.combinationFreq[hash][0] += 1;
            this.combinationFreq[hash][classHash] += 1;
        }

        // Reset statistics
        this.max = new double[4][0];
        this.min = new double[4][0];
        this.mean = new double[4];
        this.sd = new double[4];
        for (int i = 0; i < 4; i ++) {
            // Add 2 to size for both distance and class
            this.max[i] = new double[sampler.getAllPossibleValues().size() + 2];
            this.min[i] = new double[sampler.getAllPossibleValues().size() + 2];
            this.max[i][0] = -1.0f;
            this.min[i][0] = 2.0f;
            for (int j = 1; j < max[i].length; j++) {
                this.max[i][j] = -1.0;
                this.min[i][j] = -1.0;
            }
            this.mean[i] = 0.0f;
            this.sd[i] = 0.0f;
        }

    }

    @Override
    public NaiveMatrix copy() {
        return new NaiveMatrix(this.sampler);
    }

    @Override
    public void setDataSet(Instances dataSet) {
        this.sampler = new AllSamples(dataSet);
    }

    @Override
    public void setSampler(AbstractSampler sampler) {
        this.sampler = new AllSamples(sampler.getDataSet());
    }

    @Override
    public double findPv(double[] vector) {
        return this.vectorExists(vector) ?
                (double)this.combinationFreq[this.convertInstToHash(vector)][0]/
                        (double)this.sampler.getDataSet().size() : 0.0f;
    }

    public double findPy(double classValue) {
        int classIndex = this.convertClassToHash(classValue);
        if (classIndex == -1) return 0.0f;
        int nOccurrences = 0;
        for (int i = 0; i < this.combinationFreq.length; i++) {
            nOccurrences += this.combinationFreq[i][classIndex];
        }
        return (double)nOccurrences / (double)this.sampler.getDataSet().size();
    }

    @Override
    public double findPyGv(double classValue, Instance vector) {
        double[] dVector = Arrays.copyOfRange(vector.toDoubleArray(), 0, vector.numAttributes() - 1);
        int classIndex = this.convertClassToHash(classValue);
        return this.vectorExists(dVector) && classIndex != -1 ?
                (double)this.combinationFreq[this.convertInstToHash(dVector)][1+classIndex]/
                        (double)this.combinationFreq[this.convertInstToHash(dVector)][0]: 0.0f;
    }

    public double findPvGy(double classValue, Instance vector) {
        double[] dVector = Arrays.copyOfRange(vector.toDoubleArray(), 0, vector.numAttributes() - 1);
        int classIndex = this.convertClassToHash(classValue);
        if (!this.vectorExists(dVector) || classIndex == -1) return 0.0f;
        int nClassOccurrences = 0;
        for (int i = 0; i < this.combinationFreq.length; i++) {
            nClassOccurrences += this.combinationFreq[i][classIndex];
        }
        return (double)this.combinationFreq[this.convertInstToHash(dVector)][1+classIndex] / (double)nClassOccurrences;
    }

    @Override
    public double findDistance(PriorModel model1, PriorModel model2, AbstractSampler sample) {
        return 0.0f;
    }

    @Override
    public double findDistance(PosteriorModel model1, PosteriorModel model2, Instances sample) {
        return 0.0f;
    }

    private void findMeanSd(double[] allDist, int index) {
        this.mean[index] = 0.0f;
        for (int j = 0; j < allDist.length; j++) {
            mean[index] += allDist[j];
        }
        mean[index] /= allDist.length;
        this.sd[index] = 0.0f;
        for (int j = 0; j < allDist.length; j++) {
            sd[index] += Math.pow((allDist[j] - mean[index]), 2);
        }
        sd[index] /= allDist.length - 1;
        sd[index] = Math.sqrt(sd[3]);
    }

    public double[] findDistance(NaiveMatrix model1, NaiveMatrix model2, AbstractSampler sample) {
        Distance distanceMetric = new TotalVariation();
        int nClasses = sample.getAllClasses().size();
        double[] p;
        double[] q;
        double[] dist;
        int infoIndex;

        // Find p(y|v)
        infoIndex = 3;
        double finalPyGvDist = 0.0f;
        dist = new double[nClasses * sample.getNInstances()];
        sample.reset();
        for (int i = 0; i < sample.getNInstances(); i++) {
            Instance inst = sample.nextInstance();
            p = new double[nClasses];
            q = new double[nClasses];
            double weight = (model1.findPv(Arrays.copyOfRange(inst.toDoubleArray(), 0, inst.numAttributes() - 1)) +
                    model2.findPv(Arrays.copyOfRange(inst.toDoubleArray(), 0, inst.numAttributes() - 1))) / 2;
            for (int j = 0; j < nClasses; j++) {
                double yClass = Double.parseDouble(sample.getAllClasses().get(j));
                p[j] = model1.findPyGv(yClass, inst);
                q[j] = model2.findPyGv(yClass, inst);
                dist[(i * nClasses) + j] = distanceMetric.findDistance(new double[]{p[j]}, new double[]{q[j]}) * weight;
                if (dist[(i * nClasses) + j] > max[infoIndex][0]) max[infoIndex] = ArrayUtils.addAll(new double[]{dist[(i * nClasses) + j]}, inst.toDoubleArray());
                if (dist[(i * nClasses) + j] < min[infoIndex][0]) min[infoIndex] = ArrayUtils.addAll(new double[]{dist[(i * nClasses) + j]}, inst.toDoubleArray());
            }
            finalPyGvDist += distanceMetric.findDistance(p, q) * weight;
        }
        findMeanSd(dist, infoIndex);

        // Find p(v|y)
        infoIndex = 1;
        double finalPvGyDist = 0.0f;
        dist = new double[nClasses * sample.getNInstances()];
        for (int i = 0; i < nClasses; i++) {
            double yClass = Double.parseDouble(sample.getAllClasses().get(i));
            p = new double[sample.getNInstances()];
            q = new double[sample.getNInstances()];
            double weight = (model1.findPy(yClass) + model2.findPy(yClass)) / 2;
            sample.reset();
            for (int j = 0; j < sample.getNInstances(); j++) {
                Instance inst = sample.nextInstance();
                p[j] = model1.findPvGy(yClass, inst);
                q[j] = model2.findPvGy(yClass, inst);
                dist[(i * sample.getNInstances()) + j] = distanceMetric.findDistance(new double[]{p[j]}, new double[]{q[j]}) * weight;
                if (dist[(i * sample.getNInstances()) + j] > max[infoIndex][0]) max[infoIndex] = ArrayUtils.addAll(new double[]{dist[(i * sample.getNInstances()) + j]}, inst.toDoubleArray());
                if (dist[(i * sample.getNInstances()) + j] < min[infoIndex][0]) min[infoIndex] = ArrayUtils.addAll(new double[]{dist[(i * sample.getNInstances()) + j]}, inst.toDoubleArray());
            }
            finalPvGyDist += distanceMetric.findDistance(p, q) * weight;
        }
        findMeanSd(dist, infoIndex);

        // Find p(y)
        infoIndex = 2;
        p = new double[nClasses];
        q = new double[nClasses];
        dist = new double[nClasses];
        for (int j = 0; j < nClasses; j++) {
            double yClass = Double.parseDouble(sample.getAllClasses().get(j));
            p[j] = model1.findPy(yClass);
            q[j] = model2.findPy(yClass);
            dist[j] = distanceMetric.findDistance(new double[]{p[j]}, new double[]{q[j]});
            if (dist[j] > max[infoIndex][0]) {
                max[infoIndex][0] = dist[j];
                max[infoIndex][1] = yClass;
            }
            if (dist[j] < min[infoIndex][0]) {
                min[infoIndex][0] = dist[j];
                min[infoIndex][1] = yClass;
            }
        }
        double pyDist = distanceMetric.findDistance(p, q);
        findMeanSd(dist, infoIndex);

        // Find p(v)
        infoIndex = 0;
        p = new double[sample.getNInstances()];
        q = new double[sample.getNInstances()];
        dist = new double[sample.getNInstances()];
        sample.reset();
        for (int i = 0; i < sample.getNInstances(); i++) {
            Instance inst = sample.nextInstance();
            p[i] = model1.findPv(Arrays.copyOfRange(inst.toDoubleArray(), 0, inst.numAttributes() - 1));
            q[i] = model2.findPv(Arrays.copyOfRange(inst.toDoubleArray(), 0, inst.numAttributes() - 1));
            dist[i] = distanceMetric.findDistance(new double[]{p[i]}, new double[]{q[i]});
            if (dist[i] > max[infoIndex][0]) {
                max[infoIndex] = ArrayUtils.addAll(new double[]{dist[i]}, inst.toDoubleArray());
                max[infoIndex][max[infoIndex].length - 1] = -1.0;
            }
            if (dist[i] < min[infoIndex][0]) {
                min[infoIndex] = ArrayUtils.addAll(new double[]{dist[i]}, inst.toDoubleArray());
                min[infoIndex][min[infoIndex].length - 1] = -1.0;
            }
        }
        double pvDist = distanceMetric.findDistance(p, q) * sampler.getMagnitudeScale();
        findMeanSd(dist, infoIndex);

        return new double[]{pvDist, finalPvGyDist, pyDist, finalPyGvDist};
    }

    public double[][] getMax(){
        return this.max;
    }

    public double[][] getMin(){
        return this.min;
    }

    public double[] getMean() {
        return this.mean;
    }

    public double[] getSd() {
        return this.sd;
    }

    private boolean vectorExists(double[] vector) {
        ArrayList<ArrayList<String>> attributesValues = this.sampler.getAllPossibleValues();
        boolean exists = true;
        for (int attributeIndex = 0; attributeIndex < vector.length; attributeIndex++) {
            exists = exists && attributesValues.get(attributeIndex).contains(Double.toString(vector[attributeIndex]));
        }
        return exists && this.combinationFreq[this.convertInstToHash(vector)][0] > 0.0f ;
    }

    private int convertInstToHash(double[] instance) {
        int hash = 0;
        for (int i = 0; i < instance.length; i++) {
            int current_hash = 1;
            for (int j = i - 1; j >= 0; j--) {
                current_hash *= this.sampler.getAllPossibleValues().get(j).size();
            }
            current_hash *= this.sampler.getAllPossibleValues().get(i).indexOf(Double.toString(instance[i]));
            hash += current_hash;
        }
        return hash;
    }

    private double[] convertHashToInst(int hash) {
        double[] instance = new double[this.sampler.getAllPossibleValues().size()];
        for (int i = instance.length - 1; i >= 0; i--) {
            int current_value = 1;
            for (int j = i - 1; j >= 0; j--) {
                current_value *= this.sampler.getAllPossibleValues().get(j).size();
            }
            current_value *= (int) instance[i];
            instance[i] = hash % current_value;
            hash = hash / current_value;
        }
        return instance;
    }

    private int convertClassToHash(double value) {
        return this.sampler.getAllClasses().indexOf(Double.toString(value));
    }
}
