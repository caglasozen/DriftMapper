package main.models;

import org.apache.commons.lang3.ArrayUtils;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * Created by loongkuan on 11/05/16.
 **/
public class NaiveMatrix {
    private Instances allInstances;
    private int[][] frequencyMatrix;
    private int[] instanceSum;
    private int[] classSum;
    private int sampleSize;

    public NaiveMatrix(Instances dataSet) {
        // Get sample instance
        this.allInstances = new Instances(dataSet);

        // Get number of classes
        int nClasses = dataSet.classAttribute().numValues();

        // Get number hashes/combinations
        int nCombinations = 1;
        for (int i = 0; i < dataSet.numAttributes(); i++) {
            if (i != dataSet.classIndex()) {
                nCombinations *= dataSet.attribute(i).numValues();
            }
        }

        // Save the sample size
        this.sampleSize = dataSet.size();

        // Populate the Frequency Matrix and Sum tables
        frequencyMatrix = new int[nCombinations][nClasses];
        instanceSum = new int[nCombinations];
        classSum = new int[nClasses];
        System.out.println("Populating Frequency Matrix");
        for (Instance inst : dataSet) {
            int instHash = convertInstToHash(inst);
            int classHash = convertClassToHash(inst);
            // Instance Sum
            this.instanceSum[instHash] += 1;
            // Class Sum
            this.classSum[classHash] += 1;
            // Frequency Matrix
            this.frequencyMatrix[instHash][classHash] += 1;
        }
        System.out.println("Done Populating Frequency Matrix");
    }

    public Instances getAllInstances() {
        return allInstances;
    }

    private static int convertInstToHash(Instance instance) {
        int hash = 0;
        for (int i = 0; i < instance.numAttributes(); i++) {
            // Do not include the class in hash
            if (i != instance.classIndex()) {
                int current_hash = 1;
                // Get weight of current digit
                for (int j = 0; j < i; j++) {
                    // Do not include class in hash
                    if (j != instance.classIndex())
                        current_hash *= instance.attribute(j).numValues();
                }
                // Multiply weight of current digit with value of current digit
                current_hash *= (int)instance.value(i);
                // Add weighted value of current digit to hash
                hash += current_hash;
            }
        }
        return hash;
    }

    private static boolean isPartialInstance(Instance instance) {
        return instance.hasMissingValue();
    }

    private static ArrayList<Integer> convertPartialInstToHashes(Instance instance) {
        return(convertPartialInstToHashes(instance, 0, new ArrayList<>()));
    }

    private static ArrayList<Integer> convertPartialInstToHashes(Instance instance,
                                                                 int currentAttributeIndex,
                                                                 ArrayList<Integer> hashesSoFar) {
        if (currentAttributeIndex == instance.numAttributes() - 1) {
            hashesSoFar.add(convertInstToHash(instance));
            return(hashesSoFar);
        }
        else if (instance.isMissing(currentAttributeIndex)) {
            for (int attVal = 0; attVal < instance.attribute(currentAttributeIndex).numValues(); attVal++) {
                instance.setValue(currentAttributeIndex, (float)attVal);
                hashesSoFar = convertPartialInstToHashes(instance, currentAttributeIndex + 1, hashesSoFar);
            }
            return(hashesSoFar);
        }
        else {
            hashesSoFar = convertPartialInstToHashes(instance, currentAttributeIndex + 1, hashesSoFar);
            return(hashesSoFar);
        }
    }

    private static int convertClassToHash(Instance instance) {
        return (int)instance.classValue();
    }

    private boolean vectorExists(Instance instance) {
        boolean exists = this.allInstances.checkInstance(instance);
        for (int i = 0; i < instance.numAttributes(); i++) {
            exists = exists && instance.value(i) < this.allInstances.attribute(i).numValues();
        }
        return exists && this.frequencyMatrix[convertInstToHash(instance)][0] > 0.0f ;
    }

    private boolean partialVectorExists(Instance instance) {
        if (!this.allInstances.checkInstance(instance)) return false;
        for (int i = 0; i < instance.numAttributes(); i++) {
            if (!instance.isMissing(i) && instance.value(i) >= this.allInstances.attribute(i).numValues()) return false;
        }
        return true;
    }

    private int getPartialInstanceFrequency(Instance instance) {
        return this.getPartialInstanceFrequency(instance, false);
    }
    private int getPartialInstanceFrequency(Instance instance, boolean classSpecific) {
        if (!partialVectorExists(instance)) return 0;
        int totalFrequency = 0;
        ArrayList<Integer> instanceHashes = convertPartialInstToHashes(instance);
        int classHash = convertClassToHash(instance);
        for (int hash : instanceHashes) {
            totalFrequency += classSpecific ? this.frequencyMatrix[hash][classHash] : this.instanceSum[hash];
        }
        return totalFrequency;
    }

    public double findPv(Instance instance) {
        return (double)this.getPartialInstanceFrequency(instance) / (double)this.sampleSize;
    }

    public double findPy(Instance instance) {
        int classHash= convertClassToHash(instance);
        return classHash != -1 ? (double)this.classSum[classHash] / (double)this.sampleSize : 0.0f;
    }

    public double findPyGv(Instance instance) {
        int covariateFrequency = this.getPartialInstanceFrequency(instance);
        return covariateFrequency > 0 && convertClassToHash(instance) != -1 ?
                (double)this.getPartialInstanceFrequency(instance, true) / (double)covariateFrequency : 0.0f;
    }

    public double findPvGy(Instance instance) {
        if (convertClassToHash(instance) == -1) return 0.0f;
        int classFrequency = this.classSum[convertClassToHash(instance)];
        return classFrequency > 0 ?
                (double) this.getPartialInstanceFrequency(instance, true) / (double) classFrequency : 0.0f;
    }

    /*
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
                if (dist[(i * nClasses) + j] <= min[infoIndex][0]) min[infoIndex] = ArrayUtils.addAll(new double[]{dist[(i * nClasses) + j]}, inst.toDoubleArray());
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
    */


}
