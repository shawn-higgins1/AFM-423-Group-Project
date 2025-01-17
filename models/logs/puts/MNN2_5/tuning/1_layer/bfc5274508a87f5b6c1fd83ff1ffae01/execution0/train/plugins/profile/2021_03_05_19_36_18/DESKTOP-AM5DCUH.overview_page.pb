�	�FY��|5@�FY��|5@!�FY��|5@	�O�~�?�O�~�?!�O�~�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�FY��|5@��~���2@1��lY�.�?A��&��?I�G���@Y46<��?*	gffff�M@2U
Iterator::Model::ParallelMapV2?�ܵ�|�?!�K�k&;@)?�ܵ�|�?1�K�k&;@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��y�):�?!�)Q>@)�Pk�w�?1�xh��p7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeaty�&1��?!���H؛7@)��@��ǈ?1��H؛g4@:Preprocessing2F
Iterator::Model�z6�>�?!��e��#C@)F%u�{?1�w{B&@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�q����o?!�2j�N@)�q����o?1�2j�N@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip46<��?!9�++�N@)���_vOn?1�qx5�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!uR��	@)ŏ1w-!_?1uR��	@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�ݓ��Z�?!�nO�0�?@)/n��R?1_������?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�O�~�?I�܍X@Q'��i��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��~���2@��~���2@!��~���2@      ��!       "	��lY�.�?��lY�.�?!��lY�.�?*      ��!       2	��&��?��&��?!��&��?:	�G���@�G���@!�G���@B      ��!       J	46<��?46<��?!46<��?R      ��!       Z	46<��?46<��?!46<��?b      ��!       JGPUY�O�~�?b q�܍X@y'��i��?�"5
sequential/dense/MatMulMatMul&\&rʬ�?!&\&rʬ�?0"C
%gradient_tape/sequential/dense/MatMulMatMul[nwe�'�?!@��kMj�?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�I�~�1�?!�7�����?"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�I�~�1�?!Ŝ՗�?"7
sequential/dense_1/MatMulMatMul�I�~�1�?!Znwe�'�?0"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam[nwe�'�?!%\&rʬ�?"!
Adam/PowPow[nwe�'�?!�I�~�1�?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad[nwe�'�?!�7�����?"O
5gradient_tape/mean_squared_error/weighted_loss/Tile_1Tile~�Ā��?!����6�?"E
'gradient_tape/sequential/dense_1/MatMulMatMul~�Ā��?!ͫW^O[�?0Q      Y@Y��/Ċ�0@a�	�N]�T@q���XO�U@yn���5~�?"�
both�Your program is POTENTIALLY input-bound because 87.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�87.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 