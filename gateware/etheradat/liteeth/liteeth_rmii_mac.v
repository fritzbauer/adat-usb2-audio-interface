/* Machine-generated using Migen */
module liteeth_rmii_mac(
	output rmii_tx_en,
	output [1:0] rmii_txd,
	input rmii_crs_dv,
	input [1:0] rmii_rxd,
	input eth_tx_clk,
	input eth_tx_rst,
	input eth_rx_clk,
	input eth_rx_rst,
	input user_sink_valid,
	output user_sink_ready,
	input user_sink_last,
	input [7:0] user_sink_payload_data,
	output user_source_valid,
	input user_source_ready,
	output user_source_last,
	output [7:0] user_source_payload_data,
	output user_source_payload_error,
	input sys_clk,
	input sys_rst
);

reg user_sink_first = 1'd0;
wire user_sink_payload_be;
wire user_sink_payload_error;
wire user_source_first;
wire user_source_payload_be;
wire rmiitxshim_sink_valid;
wire rmiitxshim_sink_ready;
wire rmiitxshim_sink_first;
wire rmiitxshim_sink_last;
wire [7:0] rmiitxshim_sink_payload_data;
wire rmiitxshim_sink_payload_be;
wire rmiitxshim_sink_payload_error;
reg [7:0] rmiitxshim_shift = 8'd0;
reg [2:0] rmiitxshim_count = 3'd0;
wire rmiirxshim_source_valid;
wire rmiirxshim_source_ready;
reg rmiirxshim_source_first = 1'd0;
wire rmiirxshim_source_last;
wire [7:0] rmiirxshim_source_payload_data;
wire rmiirxshim_source_payload_be;
reg rmiirxshim_source_payload_error = 1'd0;
wire rmiirxshim_crs_dv;
wire [1:0] rmiirxshim_rx_data;
wire rmiirxshim_crs_first;
wire rmiirxshim_crs_last;
reg rmiirxshim_crs_run = 1'd0;
reg rmiirxshim_crs_dv_d = 1'd0;
reg [1:0] rmiirxshim_d0 = 2'd0;
reg [1:0] rmiirxshim_d1 = 2'd0;
reg rmiirxshim_e0 = 1'd0;
reg rmiirxshim_e1 = 1'd0;
wire rmiirxshim_cap;
reg [1:0] rmiirxshim_mode = 2'd0;
reg rmiirxshim_f0_valid = 1'd0;
reg [7:0] rmiirxshim_f0_data = 8'd0;
reg rmiirxshim_f0_last = 1'd0;
reg rmiirxshim_f1_valid = 1'd0;
reg [7:0] rmiirxshim_f1_data = 8'd0;
reg rmiirxshim_f1_last = 1'd0;
reg [7:0] rmiirxshim_sreg = 8'd0;
wire rmiirxshim_sfd_hit;
reg [2:0] rmiirxshim_pre_cnt = 3'd0;
wire [7:0] rmiirxshim_pre_byte;
reg [7:0] rmiirxshim_shift = 8'd0;
reg [1:0] rmiirxshim_bits = 2'd0;
wire rmiirxshim_consume;
wire [7:0] rmiirxshim_new_byte;
wire rmiirxshim_pushing;
wire sink_valid;
wire sink_ready;
wire sink_first;
wire sink_last;
wire [7:0] sink_payload_data;
wire sink_payload_be;
wire sink_payload_error;
wire source_valid;
wire source_ready;
wire source_first;
wire source_last;
wire [7:0] source_payload_data;
wire source_payload_be;
wire source_payload_error;
wire cdc_rst;
wire tx_padding_sink_valid;
reg tx_padding_sink_ready;
wire tx_padding_sink_first;
wire tx_padding_sink_last;
wire [7:0] tx_padding_sink_payload_data;
wire tx_padding_sink_payload_be;
wire tx_padding_sink_payload_error;
reg tx_padding_source_valid;
wire tx_padding_source_ready;
reg tx_padding_source_first;
reg tx_padding_source_last;
reg [7:0] tx_padding_source_payload_data;
reg tx_padding_source_payload_be;
reg tx_padding_source_payload_error;
reg [15:0] tx_padding_counter = 16'd0;
wire tx_padding_counter_done;
wire tx_crc_sink_valid;
reg tx_crc_sink_ready;
wire tx_crc_sink_first;
wire tx_crc_sink_last;
wire [7:0] tx_crc_sink_payload_data;
wire tx_crc_sink_payload_be;
wire tx_crc_sink_payload_error;
reg tx_crc_source_valid;
wire tx_crc_source_ready;
reg tx_crc_source_first;
reg tx_crc_source_last;
reg [7:0] tx_crc_source_payload_data;
reg tx_crc_source_payload_be;
reg tx_crc_source_payload_error;
reg [31:0] tx_crc_crc_packet = 32'd0;
reg tx_crc_be0 = 1'd0;
wire [7:0] tx_crc_data0;
wire tx_crc_be1;
reg [31:0] tx_crc_value;
reg tx_crc_error;
wire [7:0] tx_crc_data1;
wire [31:0] tx_crc_crc_prev;
reg [31:0] tx_crc_crc_next;
reg [31:0] tx_crc_reg = 32'd4294967295;
reg tx_crc_ce;
reg tx_crc_reset;
reg [1:0] tx_crc_cnt = 2'd3;
wire tx_crc_cnt_done;
reg tx_crc_is_ongoing0;
reg tx_crc_is_ongoing1;
wire tx_crc_sink_sink_valid;
wire tx_crc_sink_sink_ready;
wire tx_crc_sink_sink_first;
wire tx_crc_sink_sink_last;
wire [7:0] tx_crc_sink_sink_payload_data;
wire tx_crc_sink_sink_payload_be;
wire tx_crc_sink_sink_payload_error;
wire tx_crc_source_source_valid;
wire tx_crc_source_source_ready;
wire tx_crc_source_source_first;
wire tx_crc_source_source_last;
wire [7:0] tx_crc_source_source_payload_data;
wire tx_crc_source_source_payload_be;
wire tx_crc_source_source_payload_error;
wire tx_crc_pipe_valid_sink_valid;
wire tx_crc_pipe_valid_sink_ready;
wire tx_crc_pipe_valid_sink_first;
wire tx_crc_pipe_valid_sink_last;
wire [7:0] tx_crc_pipe_valid_sink_payload_data;
wire tx_crc_pipe_valid_sink_payload_be;
wire tx_crc_pipe_valid_sink_payload_error;
reg tx_crc_pipe_valid_source_valid = 1'd0;
wire tx_crc_pipe_valid_source_ready;
reg tx_crc_pipe_valid_source_first = 1'd0;
reg tx_crc_pipe_valid_source_last = 1'd0;
reg [7:0] tx_crc_pipe_valid_source_payload_data = 8'd0;
reg tx_crc_pipe_valid_source_payload_be = 1'd0;
reg tx_crc_pipe_valid_source_payload_error = 1'd0;
wire tx_preamble_sink_valid;
reg tx_preamble_sink_ready;
wire tx_preamble_sink_first;
wire tx_preamble_sink_last;
wire [7:0] tx_preamble_sink_payload_data;
wire tx_preamble_sink_payload_be;
wire tx_preamble_sink_payload_error;
reg tx_preamble_source_valid;
wire tx_preamble_source_ready;
reg tx_preamble_source_first;
reg tx_preamble_source_last;
reg [7:0] tx_preamble_source_payload_data;
reg tx_preamble_source_payload_be;
reg tx_preamble_source_payload_error;
reg [63:0] tx_preamble_preamble = 64'd15372286728091293013;
reg [2:0] tx_preamble_count = 3'd0;
wire txdatapath_sink_sink_valid;
wire txdatapath_sink_sink_ready;
wire txdatapath_sink_sink_first;
wire txdatapath_sink_sink_last;
wire [7:0] txdatapath_sink_sink_payload_data;
wire txdatapath_sink_sink_payload_be;
wire txdatapath_sink_sink_payload_error;
wire txdatapath_source_source_valid;
wire txdatapath_source_source_ready;
wire txdatapath_source_source_first;
wire txdatapath_source_source_last;
wire [7:0] txdatapath_source_source_payload_data;
wire txdatapath_source_source_payload_be;
wire txdatapath_source_source_payload_error;
wire txdatapath_ext_rst_to;
wire txdatapath_write_ext_rst;
wire txdatapath_write_rst;
wire txdatapath_cdc_sink_valid;
wire txdatapath_cdc_sink_ready;
wire txdatapath_cdc_sink_first;
wire txdatapath_cdc_sink_last;
wire [7:0] txdatapath_cdc_sink_payload_data;
wire txdatapath_cdc_sink_payload_be;
wire txdatapath_cdc_sink_payload_error;
wire txdatapath_cdc_source_valid;
wire txdatapath_cdc_source_ready;
wire txdatapath_cdc_source_first;
wire txdatapath_cdc_source_last;
wire [7:0] txdatapath_cdc_source_payload_data;
wire txdatapath_cdc_source_payload_be;
wire txdatapath_cdc_source_payload_error;
wire txdatapath_cdc_asyncfifo_we;
wire txdatapath_cdc_asyncfifo_writable;
wire txdatapath_cdc_asyncfifo_re;
wire txdatapath_cdc_asyncfifo_readable;
wire [11:0] txdatapath_cdc_asyncfifo_din;
wire [11:0] txdatapath_cdc_asyncfifo_dout;
wire txdatapath_cdc_graycounter0_ce;
(* no_retiming = "true" *) reg [5:0] txdatapath_cdc_graycounter0_q = 6'd0;
wire [5:0] txdatapath_cdc_graycounter0_q_next;
reg [5:0] txdatapath_cdc_graycounter0_q_binary = 6'd0;
reg [5:0] txdatapath_cdc_graycounter0_q_next_binary;
wire txdatapath_cdc_graycounter1_ce;
(* no_retiming = "true" *) reg [5:0] txdatapath_cdc_graycounter1_q = 6'd0;
wire [5:0] txdatapath_cdc_graycounter1_q_next;
reg [5:0] txdatapath_cdc_graycounter1_q_binary = 6'd0;
reg [5:0] txdatapath_cdc_graycounter1_q_next_binary;
wire [5:0] txdatapath_cdc_produce_rdomain;
wire [5:0] txdatapath_cdc_consume_wdomain;
wire [4:0] txdatapath_cdc_wrport_adr;
wire txdatapath_cdc_wrport_we;
wire [11:0] txdatapath_cdc_wrport_dat_w;
wire [4:0] txdatapath_cdc_rdport_adr;
wire [11:0] txdatapath_cdc_rdport_dat_r;
wire [7:0] txdatapath_cdc_fifo_in_payload_data;
wire txdatapath_cdc_fifo_in_payload_be;
wire txdatapath_cdc_fifo_in_payload_error;
wire txdatapath_cdc_fifo_in_first;
wire txdatapath_cdc_fifo_in_last;
wire [7:0] txdatapath_cdc_fifo_out_payload_data;
wire txdatapath_cdc_fifo_out_payload_be;
wire txdatapath_cdc_fifo_out_payload_error;
wire txdatapath_cdc_fifo_out_first;
wire txdatapath_cdc_fifo_out_last;
wire txdatapath_cdc_reset_write;
wire txdatapath_cdc_reset_read;
wire tx_packet_fifo_sink_sink_valid;
wire tx_packet_fifo_sink_sink_ready;
wire tx_packet_fifo_sink_sink_first;
wire tx_packet_fifo_sink_sink_last;
wire [7:0] tx_packet_fifo_sink_sink_payload_data;
wire tx_packet_fifo_sink_sink_payload_be;
wire tx_packet_fifo_sink_sink_payload_error;
wire tx_packet_fifo_source_source_valid;
wire tx_packet_fifo_source_source_ready;
reg tx_packet_fifo_source_source_first;
wire tx_packet_fifo_source_source_last;
wire [7:0] tx_packet_fifo_source_source_payload_data;
wire tx_packet_fifo_source_source_payload_be;
wire tx_packet_fifo_source_source_payload_error;
wire tx_packet_fifo_payload_fifo_sink_valid;
wire tx_packet_fifo_payload_fifo_sink_ready;
wire tx_packet_fifo_payload_fifo_sink_first;
wire tx_packet_fifo_payload_fifo_sink_last;
wire [7:0] tx_packet_fifo_payload_fifo_sink_payload_data;
wire tx_packet_fifo_payload_fifo_sink_payload_be;
wire tx_packet_fifo_payload_fifo_sink_payload_error;
wire tx_packet_fifo_payload_fifo_source_valid;
wire tx_packet_fifo_payload_fifo_source_ready;
wire tx_packet_fifo_payload_fifo_source_first;
wire tx_packet_fifo_payload_fifo_source_last;
wire [7:0] tx_packet_fifo_payload_fifo_source_payload_data;
wire tx_packet_fifo_payload_fifo_source_payload_be;
wire tx_packet_fifo_payload_fifo_source_payload_error;
wire tx_packet_fifo_payload_fifo_re;
reg tx_packet_fifo_payload_fifo_readable = 1'd0;
wire tx_packet_fifo_payload_fifo_syncfifo_we;
wire tx_packet_fifo_payload_fifo_syncfifo_writable;
wire tx_packet_fifo_payload_fifo_syncfifo_re;
wire tx_packet_fifo_payload_fifo_syncfifo_readable;
wire [11:0] tx_packet_fifo_payload_fifo_syncfifo_din;
wire [11:0] tx_packet_fifo_payload_fifo_syncfifo_dout;
reg [11:0] tx_packet_fifo_payload_fifo_level0 = 12'd0;
reg tx_packet_fifo_payload_fifo_replace = 1'd0;
reg [10:0] tx_packet_fifo_payload_fifo_produce = 11'd0;
reg [10:0] tx_packet_fifo_payload_fifo_consume = 11'd0;
reg [10:0] tx_packet_fifo_payload_fifo_wrport_adr;
wire tx_packet_fifo_payload_fifo_wrport_we;
wire [11:0] tx_packet_fifo_payload_fifo_wrport_dat_w;
wire tx_packet_fifo_payload_fifo_do_read;
wire [10:0] tx_packet_fifo_payload_fifo_rdport_adr;
wire [11:0] tx_packet_fifo_payload_fifo_rdport_dat_r;
wire tx_packet_fifo_payload_fifo_rdport_re;
wire [11:0] tx_packet_fifo_payload_fifo_level1;
wire [7:0] tx_packet_fifo_payload_fifo_fifo_in_payload_data;
wire tx_packet_fifo_payload_fifo_fifo_in_payload_be;
wire tx_packet_fifo_payload_fifo_fifo_in_payload_error;
wire tx_packet_fifo_payload_fifo_fifo_in_first;
wire tx_packet_fifo_payload_fifo_fifo_in_last;
wire [7:0] tx_packet_fifo_payload_fifo_fifo_out_payload_data;
wire tx_packet_fifo_payload_fifo_fifo_out_payload_be;
wire tx_packet_fifo_payload_fifo_fifo_out_payload_error;
wire tx_packet_fifo_payload_fifo_fifo_out_first;
wire tx_packet_fifo_payload_fifo_fifo_out_last;
wire tx_packet_fifo_param_fifo_sink_valid;
wire tx_packet_fifo_param_fifo_sink_ready;
reg tx_packet_fifo_param_fifo_sink_first = 1'd0;
reg tx_packet_fifo_param_fifo_sink_last = 1'd0;
reg tx_packet_fifo_param_fifo_sink_param_dummy = 1'd0;
wire tx_packet_fifo_param_fifo_source_valid;
wire tx_packet_fifo_param_fifo_source_ready;
wire tx_packet_fifo_param_fifo_source_first;
wire tx_packet_fifo_param_fifo_source_last;
wire tx_packet_fifo_param_fifo_source_param_dummy;
wire tx_packet_fifo_param_fifo_re;
reg tx_packet_fifo_param_fifo_readable = 1'd0;
wire tx_packet_fifo_param_fifo_syncfifo_we;
wire tx_packet_fifo_param_fifo_syncfifo_writable;
wire tx_packet_fifo_param_fifo_syncfifo_re;
wire tx_packet_fifo_param_fifo_syncfifo_readable;
wire [2:0] tx_packet_fifo_param_fifo_syncfifo_din;
wire [2:0] tx_packet_fifo_param_fifo_syncfifo_dout;
reg [2:0] tx_packet_fifo_param_fifo_level0 = 3'd0;
reg tx_packet_fifo_param_fifo_replace = 1'd0;
reg [2:0] tx_packet_fifo_param_fifo_produce = 3'd0;
reg [2:0] tx_packet_fifo_param_fifo_consume = 3'd0;
reg [2:0] tx_packet_fifo_param_fifo_wrport_adr;
wire tx_packet_fifo_param_fifo_wrport_we;
wire [2:0] tx_packet_fifo_param_fifo_wrport_dat_w;
wire tx_packet_fifo_param_fifo_do_read;
wire [2:0] tx_packet_fifo_param_fifo_rdport_adr;
wire [2:0] tx_packet_fifo_param_fifo_rdport_dat_r;
wire tx_packet_fifo_param_fifo_rdport_re;
wire [2:0] tx_packet_fifo_param_fifo_level1;
wire tx_packet_fifo_param_fifo_fifo_in_param_dummy;
wire tx_packet_fifo_param_fifo_fifo_in_first;
wire tx_packet_fifo_param_fifo_fifo_in_last;
wire tx_packet_fifo_param_fifo_fifo_out_param_dummy;
wire tx_packet_fifo_param_fifo_fifo_out_first;
wire tx_packet_fifo_param_fifo_fifo_out_last;
wire tx_packet_fifo_reset;
wire tx_gap_sink_valid;
reg tx_gap_sink_ready;
wire tx_gap_sink_first;
wire tx_gap_sink_last;
wire [7:0] tx_gap_sink_payload_data;
wire tx_gap_sink_payload_be;
wire tx_gap_sink_payload_error;
reg tx_gap_source_valid;
wire tx_gap_source_ready;
reg tx_gap_source_first;
reg tx_gap_source_last;
reg [7:0] tx_gap_source_payload_data;
reg tx_gap_source_payload_be;
reg tx_gap_source_payload_error;
reg [3:0] tx_gap_counter = 4'd0;
wire tx_gap_reset;
reg [31:0] preamble_errors_status = 32'd0;
reg [31:0] crc_errors_status = 32'd0;
reg [31:0] drops_status = 32'd0;
wire rx_packet_fifo_sink_sink_valid;
wire rx_packet_fifo_sink_sink_ready;
wire rx_packet_fifo_sink_sink_first;
wire rx_packet_fifo_sink_sink_last;
wire [7:0] rx_packet_fifo_sink_sink_payload_data;
wire rx_packet_fifo_sink_sink_payload_be;
wire rx_packet_fifo_sink_sink_payload_error;
wire rx_packet_fifo_source_source_valid;
wire rx_packet_fifo_source_source_ready;
wire rx_packet_fifo_source_source_first;
wire rx_packet_fifo_source_source_last;
wire [7:0] rx_packet_fifo_source_source_payload_data;
wire rx_packet_fifo_source_source_payload_be;
wire rx_packet_fifo_source_source_payload_error;
reg rx_packet_fifo_discard = 1'd0;
wire rx_packet_fifo_drop;
wire rx_packet_fifo_param_fifo_sink_valid;
wire rx_packet_fifo_param_fifo_sink_ready;
reg rx_packet_fifo_param_fifo_sink_first = 1'd0;
wire rx_packet_fifo_param_fifo_sink_last;
reg rx_packet_fifo_param_fifo_sink_param_dummy = 1'd0;
wire rx_packet_fifo_param_fifo_source_valid;
wire rx_packet_fifo_param_fifo_source_ready;
wire rx_packet_fifo_param_fifo_source_first;
wire rx_packet_fifo_param_fifo_source_last;
wire rx_packet_fifo_param_fifo_source_param_dummy;
wire rx_packet_fifo_param_fifo_syncfifo_we;
wire rx_packet_fifo_param_fifo_syncfifo_writable;
wire rx_packet_fifo_param_fifo_syncfifo_re;
wire rx_packet_fifo_param_fifo_syncfifo_readable;
wire [2:0] rx_packet_fifo_param_fifo_syncfifo_din;
wire [2:0] rx_packet_fifo_param_fifo_syncfifo_dout;
reg [2:0] rx_packet_fifo_param_fifo_level = 3'd0;
reg rx_packet_fifo_param_fifo_replace = 1'd0;
reg [1:0] rx_packet_fifo_param_fifo_produce = 2'd0;
reg [1:0] rx_packet_fifo_param_fifo_consume = 2'd0;
reg [1:0] rx_packet_fifo_param_fifo_wrport_adr;
wire rx_packet_fifo_param_fifo_wrport_we;
wire [2:0] rx_packet_fifo_param_fifo_wrport_dat_w;
wire rx_packet_fifo_param_fifo_do_read;
wire [1:0] rx_packet_fifo_param_fifo_rdport_adr;
wire [2:0] rx_packet_fifo_param_fifo_rdport_dat_r;
wire rx_packet_fifo_param_fifo_fifo_in_param_dummy;
wire rx_packet_fifo_param_fifo_fifo_in_first;
wire rx_packet_fifo_param_fifo_fifo_in_last;
wire rx_packet_fifo_param_fifo_fifo_out_param_dummy;
wire rx_packet_fifo_param_fifo_fifo_out_first;
wire rx_packet_fifo_param_fifo_fifo_out_last;
wire [10:0] rx_packet_fifo_wr_port_adr;
wire [11:0] rx_packet_fifo_wr_port_dat_r;
wire rx_packet_fifo_wr_port_we;
wire [11:0] rx_packet_fifo_wr_port_dat_w;
wire [10:0] rx_packet_fifo_rd_port_adr;
wire [11:0] rx_packet_fifo_rd_port_dat_r;
wire rx_packet_fifo_rd_port_re;
reg [11:0] rx_packet_fifo_wr_ptr = 12'd0;
reg [11:0] rx_packet_fifo_commit_ptr = 12'd0;
reg [11:0] rx_packet_fifo_rd_ptr = 12'd0;
reg rx_packet_fifo_out_valid = 1'd0;
wire [11:0] rx_packet_fifo_used;
wire rx_packet_fifo_full;
reg rx_packet_fifo_overflow = 1'd0;
reg rx_packet_fifo_invalid = 1'd0;
wire rx_packet_fifo_commit;
wire rx_packet_fifo_fetch;
wire i;
wire o;
reg toggle_i = 1'd0;
wire toggle_o;
reg toggle_o_r = 1'd0;
wire rx_cdc_sink_sink_valid;
wire rx_cdc_sink_sink_ready;
wire rx_cdc_sink_sink_first;
wire rx_cdc_sink_sink_last;
wire [7:0] rx_cdc_sink_sink_payload_data;
wire rx_cdc_sink_sink_payload_be;
wire rx_cdc_sink_sink_payload_error;
wire rx_cdc_source_source_valid;
wire rx_cdc_source_source_ready;
wire rx_cdc_source_source_first;
wire rx_cdc_source_source_last;
wire [7:0] rx_cdc_source_source_payload_data;
wire rx_cdc_source_source_payload_be;
wire rx_cdc_source_source_payload_error;
wire rx_cdc_ext_rst_to;
wire rx_cdc_write_ext_rst;
wire rx_cdc_write_rst;
wire rx_cdc_cdc_sink_valid;
wire rx_cdc_cdc_sink_ready;
wire rx_cdc_cdc_sink_first;
wire rx_cdc_cdc_sink_last;
wire [7:0] rx_cdc_cdc_sink_payload_data;
wire rx_cdc_cdc_sink_payload_be;
wire rx_cdc_cdc_sink_payload_error;
wire rx_cdc_cdc_source_valid;
wire rx_cdc_cdc_source_ready;
wire rx_cdc_cdc_source_first;
wire rx_cdc_cdc_source_last;
wire [7:0] rx_cdc_cdc_source_payload_data;
wire rx_cdc_cdc_source_payload_be;
wire rx_cdc_cdc_source_payload_error;
wire rx_cdc_cdc_asyncfifo_we;
wire rx_cdc_cdc_asyncfifo_writable;
wire rx_cdc_cdc_asyncfifo_re;
wire rx_cdc_cdc_asyncfifo_readable;
wire [11:0] rx_cdc_cdc_asyncfifo_din;
wire [11:0] rx_cdc_cdc_asyncfifo_dout;
wire rx_cdc_cdc_graycounter0_ce;
(* no_retiming = "true" *) reg [5:0] rx_cdc_cdc_graycounter0_q = 6'd0;
wire [5:0] rx_cdc_cdc_graycounter0_q_next;
reg [5:0] rx_cdc_cdc_graycounter0_q_binary = 6'd0;
reg [5:0] rx_cdc_cdc_graycounter0_q_next_binary;
wire rx_cdc_cdc_graycounter1_ce;
(* no_retiming = "true" *) reg [5:0] rx_cdc_cdc_graycounter1_q = 6'd0;
wire [5:0] rx_cdc_cdc_graycounter1_q_next;
reg [5:0] rx_cdc_cdc_graycounter1_q_binary = 6'd0;
reg [5:0] rx_cdc_cdc_graycounter1_q_next_binary;
wire [5:0] rx_cdc_cdc_produce_rdomain;
wire [5:0] rx_cdc_cdc_consume_wdomain;
wire [4:0] rx_cdc_cdc_wrport_adr;
wire rx_cdc_cdc_wrport_we;
wire [11:0] rx_cdc_cdc_wrport_dat_w;
wire [4:0] rx_cdc_cdc_rdport_adr;
wire [11:0] rx_cdc_cdc_rdport_dat_r;
wire [7:0] rx_cdc_cdc_fifo_in_payload_data;
wire rx_cdc_cdc_fifo_in_payload_be;
wire rx_cdc_cdc_fifo_in_payload_error;
wire rx_cdc_cdc_fifo_in_first;
wire rx_cdc_cdc_fifo_in_last;
wire [7:0] rx_cdc_cdc_fifo_out_payload_data;
wire rx_cdc_cdc_fifo_out_payload_be;
wire rx_cdc_cdc_fifo_out_payload_error;
wire rx_cdc_cdc_fifo_out_first;
wire rx_cdc_cdc_fifo_out_last;
wire rx_cdc_cdc_reset_write;
wire rx_cdc_cdc_reset_read;
wire rx_preamble_sink_valid;
reg rx_preamble_sink_ready;
wire rx_preamble_sink_first;
wire rx_preamble_sink_last;
wire [7:0] rx_preamble_sink_payload_data;
wire rx_preamble_sink_payload_be;
wire rx_preamble_sink_payload_error;
reg rx_preamble_source_valid;
wire rx_preamble_source_ready;
reg rx_preamble_source_first;
reg rx_preamble_source_last;
wire [7:0] rx_preamble_source_payload_data;
wire rx_preamble_source_payload_be;
reg rx_preamble_source_payload_error;
reg rx_preamble_error;
reg [63:0] rx_preamble_preamble = 64'd15372286728091293013;
wire pulsesynchronizer0_i;
wire pulsesynchronizer0_o;
reg pulsesynchronizer0_toggle_i = 1'd0;
wire pulsesynchronizer0_toggle_o;
reg pulsesynchronizer0_toggle_o_r = 1'd0;
wire liteethmaccrc32checker_sink_sink_valid;
reg liteethmaccrc32checker_sink_sink_ready;
wire liteethmaccrc32checker_sink_sink_first;
wire liteethmaccrc32checker_sink_sink_last;
wire [7:0] liteethmaccrc32checker_sink_sink_payload_data;
wire liteethmaccrc32checker_sink_sink_payload_be;
wire liteethmaccrc32checker_sink_sink_payload_error;
reg liteethmaccrc32checker_source_source_valid;
wire liteethmaccrc32checker_source_source_ready;
wire liteethmaccrc32checker_source_source_first;
reg liteethmaccrc32checker_source_source_last;
wire [7:0] liteethmaccrc32checker_source_source_payload_data;
reg liteethmaccrc32checker_source_source_payload_be;
reg liteethmaccrc32checker_source_source_payload_error;
reg liteethmaccrc32checker_error;
wire [7:0] liteethmaccrc32checker_crc_data0;
wire liteethmaccrc32checker_crc_be;
reg liteethmaccrc32checker_crc_error;
reg [7:0] liteethmaccrc32checker_crc_data1;
wire [31:0] liteethmaccrc32checker_crc_crc_prev;
reg [31:0] liteethmaccrc32checker_crc_crc_next;
reg [31:0] liteethmaccrc32checker_crc_reg = 32'd4294967295;
reg liteethmaccrc32checker_crc_ce;
reg liteethmaccrc32checker_crc_reset;
reg liteethmaccrc32checker_syncfifo_sink_valid;
wire liteethmaccrc32checker_syncfifo_sink_ready;
wire liteethmaccrc32checker_syncfifo_sink_first;
wire liteethmaccrc32checker_syncfifo_sink_last;
wire [7:0] liteethmaccrc32checker_syncfifo_sink_payload_data;
wire liteethmaccrc32checker_syncfifo_sink_payload_be;
wire liteethmaccrc32checker_syncfifo_sink_payload_error;
wire liteethmaccrc32checker_syncfifo_source_valid;
reg liteethmaccrc32checker_syncfifo_source_ready;
wire liteethmaccrc32checker_syncfifo_source_first;
wire liteethmaccrc32checker_syncfifo_source_last;
wire [7:0] liteethmaccrc32checker_syncfifo_source_payload_data;
wire liteethmaccrc32checker_syncfifo_source_payload_be;
wire liteethmaccrc32checker_syncfifo_source_payload_error;
wire liteethmaccrc32checker_syncfifo_syncfifo_we;
wire liteethmaccrc32checker_syncfifo_syncfifo_writable;
wire liteethmaccrc32checker_syncfifo_syncfifo_re;
wire liteethmaccrc32checker_syncfifo_syncfifo_readable;
wire [11:0] liteethmaccrc32checker_syncfifo_syncfifo_din;
wire [11:0] liteethmaccrc32checker_syncfifo_syncfifo_dout;
reg [2:0] liteethmaccrc32checker_syncfifo_level = 3'd0;
reg liteethmaccrc32checker_syncfifo_replace = 1'd0;
reg [2:0] liteethmaccrc32checker_syncfifo_produce = 3'd0;
reg [2:0] liteethmaccrc32checker_syncfifo_consume = 3'd0;
reg [2:0] liteethmaccrc32checker_syncfifo_wrport_adr;
wire liteethmaccrc32checker_syncfifo_wrport_we;
wire [11:0] liteethmaccrc32checker_syncfifo_wrport_dat_w;
wire liteethmaccrc32checker_syncfifo_do_read;
wire [2:0] liteethmaccrc32checker_syncfifo_rdport_adr;
wire [11:0] liteethmaccrc32checker_syncfifo_rdport_dat_r;
wire [7:0] liteethmaccrc32checker_syncfifo_fifo_in_payload_data;
wire liteethmaccrc32checker_syncfifo_fifo_in_payload_be;
wire liteethmaccrc32checker_syncfifo_fifo_in_payload_error;
wire liteethmaccrc32checker_syncfifo_fifo_in_first;
wire liteethmaccrc32checker_syncfifo_fifo_in_last;
wire [7:0] liteethmaccrc32checker_syncfifo_fifo_out_payload_data;
wire liteethmaccrc32checker_syncfifo_fifo_out_payload_be;
wire liteethmaccrc32checker_syncfifo_fifo_out_payload_error;
wire liteethmaccrc32checker_syncfifo_fifo_out_first;
wire liteethmaccrc32checker_syncfifo_fifo_out_last;
reg liteethmaccrc32checker_fifo_reset;
wire liteethmaccrc32checker_fifo_in;
wire liteethmaccrc32checker_fifo_out;
wire liteethmaccrc32checker_fifo_full;
reg liteethmaccrc32checker_is_ongoing0;
reg liteethmaccrc32checker_is_ongoing1;
reg liteethmaccrc32checker_be = 1'd0;
reg liteethmaccrc32checker_last_error = 1'd0;
wire liteethmaccrc32checker_last_word_error;
reg liteethmaccrc32checker_packet_error = 1'd0;
reg liteethmaccrc32checker_is_ongoing2;
wire bufferizeendpoints_sink_sink_valid;
wire bufferizeendpoints_sink_sink_ready;
wire bufferizeendpoints_sink_sink_first;
wire bufferizeendpoints_sink_sink_last;
wire [7:0] bufferizeendpoints_sink_sink_payload_data;
wire bufferizeendpoints_sink_sink_payload_be;
wire bufferizeendpoints_sink_sink_payload_error;
wire bufferizeendpoints_source_source_valid;
wire bufferizeendpoints_source_source_ready;
wire bufferizeendpoints_source_source_first;
wire bufferizeendpoints_source_source_last;
wire [7:0] bufferizeendpoints_source_source_payload_data;
wire bufferizeendpoints_source_source_payload_be;
wire bufferizeendpoints_source_source_payload_error;
wire bufferizeendpoints_pipe_valid_sink_valid;
wire bufferizeendpoints_pipe_valid_sink_ready;
wire bufferizeendpoints_pipe_valid_sink_first;
wire bufferizeendpoints_pipe_valid_sink_last;
wire [7:0] bufferizeendpoints_pipe_valid_sink_payload_data;
wire bufferizeendpoints_pipe_valid_sink_payload_be;
wire bufferizeendpoints_pipe_valid_sink_payload_error;
reg bufferizeendpoints_pipe_valid_source_valid = 1'd0;
wire bufferizeendpoints_pipe_valid_source_ready;
reg bufferizeendpoints_pipe_valid_source_first = 1'd0;
reg bufferizeendpoints_pipe_valid_source_last = 1'd0;
reg [7:0] bufferizeendpoints_pipe_valid_source_payload_data = 8'd0;
reg bufferizeendpoints_pipe_valid_source_payload_be = 1'd0;
reg bufferizeendpoints_pipe_valid_source_payload_error = 1'd0;
wire pulsesynchronizer1_i;
wire pulsesynchronizer1_o;
reg pulsesynchronizer1_toggle_i = 1'd0;
wire pulsesynchronizer1_toggle_o;
reg pulsesynchronizer1_toggle_o_r = 1'd0;
wire rx_padding_sink_valid;
wire rx_padding_sink_ready;
wire rx_padding_sink_first;
wire rx_padding_sink_last;
wire [7:0] rx_padding_sink_payload_data;
wire rx_padding_sink_payload_be;
wire rx_padding_sink_payload_error;
wire rx_padding_source_valid;
wire rx_padding_source_ready;
wire rx_padding_source_first;
wire rx_padding_source_last;
wire [7:0] rx_padding_source_payload_data;
wire rx_padding_source_payload_be;
reg rx_padding_source_payload_error;
reg [10:0] rx_padding_length = 11'd0;
wire [3:0] rx_padding_length_inc;
reg txdatapath_liteethmacpaddinginserter_state = 1'd0;
reg txdatapath_liteethmacpaddinginserter_next_state;
reg [15:0] tx_padding_counter_clockdomainsrenamer0_next_value;
reg tx_padding_counter_clockdomainsrenamer0_next_value_ce;
reg [1:0] txdatapath_bufferizeendpoints_state = 2'd0;
reg [1:0] txdatapath_bufferizeendpoints_next_state;
reg [31:0] tx_crc_crc_packet_clockdomainsrenamer1_next_value0;
reg tx_crc_crc_packet_clockdomainsrenamer1_next_value_ce0;
reg tx_crc_be0_clockdomainsrenamer1_next_value1;
reg tx_crc_be0_clockdomainsrenamer1_next_value_ce1;
reg [1:0] txdatapath_liteethmacpreambleinserter_state = 2'd0;
reg [1:0] txdatapath_liteethmacpreambleinserter_next_state;
reg [2:0] tx_preamble_count_clockdomainsrenamer2_next_value;
reg tx_preamble_count_clockdomainsrenamer2_next_value_ce;
reg txdatapath_resetinserter_state = 1'd0;
reg txdatapath_resetinserter_next_state;
reg [3:0] tx_gap_counter_clockdomainsrenamer3_next_value;
reg tx_gap_counter_clockdomainsrenamer3_next_value_ce;
reg rxdatapath_liteethmacpreamblechecker_state = 1'd0;
reg rxdatapath_liteethmacpreamblechecker_next_state;
reg [1:0] rxdatapath_bufferizeendpoints_state = 2'd0;
reg [1:0] rxdatapath_bufferizeendpoints_next_state;
reg liteethmaccrc32checker_be_next_value0;
reg liteethmaccrc32checker_be_next_value_ce0;
reg liteethmaccrc32checker_last_error_next_value1;
reg liteethmaccrc32checker_last_error_next_value_ce1;
wire [39:0] slice_proxy;
(* no_retiming = "true" *) reg multiregimpl00 = 1'd0;
(* no_retiming = "true" *) reg multiregimpl01 = 1'd0;
(* no_retiming = "true" *) reg multiregimpl10 = 1'd0;
(* no_retiming = "true" *) reg multiregimpl11 = 1'd0;
(* no_retiming = "true" *) reg [5:0] multiregimpl20 = 6'd0;
(* no_retiming = "true" *) reg [5:0] multiregimpl21 = 6'd0;
(* no_retiming = "true" *) reg [5:0] multiregimpl30 = 6'd0;
(* no_retiming = "true" *) reg [5:0] multiregimpl31 = 6'd0;
(* no_retiming = "true" *) reg multiregimpl40 = 1'd0;
(* no_retiming = "true" *) reg multiregimpl41 = 1'd0;
(* no_retiming = "true" *) reg multiregimpl50 = 1'd0;
(* no_retiming = "true" *) reg multiregimpl51 = 1'd0;
(* no_retiming = "true" *) reg multiregimpl60 = 1'd0;
(* no_retiming = "true" *) reg multiregimpl61 = 1'd0;
(* no_retiming = "true" *) reg [5:0] multiregimpl70 = 6'd0;
(* no_retiming = "true" *) reg [5:0] multiregimpl71 = 6'd0;
(* no_retiming = "true" *) reg [5:0] multiregimpl80 = 6'd0;
(* no_retiming = "true" *) reg [5:0] multiregimpl81 = 6'd0;
(* no_retiming = "true" *) reg multiregimpl90 = 1'd0;
(* no_retiming = "true" *) reg multiregimpl91 = 1'd0;
(* no_retiming = "true" *) reg multiregimpl100 = 1'd0;
(* no_retiming = "true" *) reg multiregimpl101 = 1'd0;

// synthesis translate_off
reg dummy_s;
initial dummy_s <= 1'd0;
// synthesis translate_on

assign user_sink_payload_be = 1'd1;
assign user_sink_payload_error = 1'd0;
assign sink_valid = user_sink_valid;
assign user_sink_ready = sink_ready;
assign sink_first = user_sink_first;
assign sink_last = user_sink_last;
assign sink_payload_data = user_sink_payload_data;
assign sink_payload_be = user_sink_payload_be;
assign sink_payload_error = user_sink_payload_error;
assign user_source_valid = source_valid;
assign source_ready = user_source_ready;
assign user_source_first = source_first;
assign user_source_last = source_last;
assign user_source_payload_data = source_payload_data;
assign user_source_payload_be = source_payload_be;
assign user_source_payload_error = source_payload_error;
assign rmii_txd = rmiitxshim_shift[1:0];
assign rmii_tx_en = (rmiitxshim_count != 1'd0);
assign rmiitxshim_sink_ready = (rmiitxshim_count <= 1'd1);
assign rmiirxshim_source_payload_be = 1'd1;
assign rmiirxshim_crs_dv = rmii_crs_dv;
assign rmiirxshim_rx_data = rmii_rxd;
assign rmiirxshim_crs_first = (rmiirxshim_crs_dv & (rmiirxshim_rx_data != 1'd0));
assign rmiirxshim_crs_last = ((~rmiirxshim_crs_dv) & (~rmiirxshim_crs_dv_d));
assign rmiirxshim_cap = (rmiirxshim_crs_first | rmiirxshim_crs_run);
assign rmiirxshim_sfd_hit = (((rmiirxshim_e1 & (rmiirxshim_mode == 1'd0)) & (rmiirxshim_sreg[5:0] == 5'd21)) & (rmiirxshim_d1 == 2'd3));
assign rmiirxshim_pre_byte = ((rmiirxshim_pre_cnt == 3'd7) ? 8'd213 : 7'd85);
assign rmiirxshim_source_valid = ((rmiirxshim_mode == 1'd1) | ((rmiirxshim_mode == 2'd2) & rmiirxshim_f0_valid));
assign rmiirxshim_source_payload_data = ((rmiirxshim_mode == 1'd1) ? rmiirxshim_pre_byte : rmiirxshim_f0_data);
assign rmiirxshim_source_last = ((rmiirxshim_mode == 2'd2) & rmiirxshim_f0_last);
assign rmiirxshim_consume = (rmiirxshim_source_valid & rmiirxshim_source_ready);
assign rmiirxshim_new_byte = {rmiirxshim_d1, rmiirxshim_shift[7:2]};
assign rmiirxshim_pushing = (((rmiirxshim_mode != 1'd0) & rmiirxshim_e1) & (rmiirxshim_bits == 2'd3));
assign cdc_rst = txdatapath_ext_rst_to;
assign tx_packet_fifo_reset = cdc_rst;
assign tx_gap_reset = cdc_rst;
assign tx_padding_counter_done = (tx_padding_counter >= 6'd59);

// synthesis translate_off
reg dummy_d;
// synthesis translate_on
always @(*) begin
	tx_padding_sink_ready <= 1'd0;
	tx_padding_source_valid <= 1'd0;
	tx_padding_source_first <= 1'd0;
	tx_padding_source_last <= 1'd0;
	tx_padding_source_payload_data <= 8'd0;
	tx_padding_source_payload_be <= 1'd0;
	tx_padding_source_payload_error <= 1'd0;
	txdatapath_liteethmacpaddinginserter_next_state <= 1'd0;
	tx_padding_counter_clockdomainsrenamer0_next_value <= 16'd0;
	tx_padding_counter_clockdomainsrenamer0_next_value_ce <= 1'd0;
	txdatapath_liteethmacpaddinginserter_next_state <= txdatapath_liteethmacpaddinginserter_state;
	case (txdatapath_liteethmacpaddinginserter_state)
		1'd1: begin
			tx_padding_source_valid <= 1'd1;
			tx_padding_source_payload_be <= 1'd1;
			if (tx_padding_counter_done) begin
				tx_padding_source_payload_be <= 1'd1;
				tx_padding_source_last <= 1'd1;
			end
			tx_padding_source_payload_data <= 1'd0;
			if ((tx_padding_source_valid & tx_padding_source_ready)) begin
				tx_padding_counter_clockdomainsrenamer0_next_value <= (tx_padding_counter + 1'd1);
				tx_padding_counter_clockdomainsrenamer0_next_value_ce <= 1'd1;
				if (tx_padding_counter_done) begin
					tx_padding_counter_clockdomainsrenamer0_next_value <= 1'd0;
					tx_padding_counter_clockdomainsrenamer0_next_value_ce <= 1'd1;
					txdatapath_liteethmacpaddinginserter_next_state <= 1'd0;
				end
			end
		end
		default: begin
			tx_padding_source_valid <= tx_padding_sink_valid;
			tx_padding_sink_ready <= tx_padding_source_ready;
			tx_padding_source_first <= tx_padding_sink_first;
			tx_padding_source_last <= tx_padding_sink_last;
			tx_padding_source_payload_data <= tx_padding_sink_payload_data;
			tx_padding_source_payload_be <= tx_padding_sink_payload_be;
			tx_padding_source_payload_error <= tx_padding_sink_payload_error;
			if (tx_padding_sink_last) begin
				if ((~tx_padding_sink_payload_be)) begin
					tx_padding_source_payload_data[7:0] <= 1'd0;
				end
				if ((~tx_padding_counter_done)) begin
					tx_padding_source_last <= 1'd0;
					tx_padding_source_payload_be <= 1'd1;
				end else begin
					if (((tx_padding_counter == 6'd59) & (tx_padding_sink_payload_be < 1'd1))) begin
						tx_padding_source_payload_be <= 1'd1;
					end
				end
			end
			if ((tx_padding_source_valid & tx_padding_source_ready)) begin
				tx_padding_counter_clockdomainsrenamer0_next_value <= (tx_padding_counter + 1'd1);
				tx_padding_counter_clockdomainsrenamer0_next_value_ce <= 1'd1;
				if (tx_padding_sink_last) begin
					if ((~tx_padding_counter_done)) begin
						txdatapath_liteethmacpaddinginserter_next_state <= 1'd1;
					end else begin
						tx_padding_counter_clockdomainsrenamer0_next_value <= 1'd0;
						tx_padding_counter_clockdomainsrenamer0_next_value_ce <= 1'd1;
					end
				end
			end
		end
	endcase
// synthesis translate_off
	dummy_d <= dummy_s;
// synthesis translate_on
end
assign tx_crc_data0 = tx_crc_sink_payload_data;
assign tx_crc_be1 = tx_crc_sink_payload_be;
assign tx_crc_cnt_done = (tx_crc_cnt == 1'd0);
assign tx_crc_sink_valid = tx_crc_source_source_valid;
assign tx_crc_source_source_ready = tx_crc_sink_ready;
assign tx_crc_sink_first = tx_crc_source_source_first;
assign tx_crc_sink_last = tx_crc_source_source_last;
assign tx_crc_sink_payload_data = tx_crc_source_source_payload_data;
assign tx_crc_sink_payload_be = tx_crc_source_source_payload_be;
assign tx_crc_sink_payload_error = tx_crc_source_source_payload_error;
assign tx_crc_data1 = tx_crc_data0;
assign tx_crc_crc_prev = tx_crc_reg;

// synthesis translate_off
reg dummy_d_1;
// synthesis translate_on
always @(*) begin
	tx_crc_value <= 32'd0;
	tx_crc_error <= 1'd0;
	if (tx_crc_be1) begin
		tx_crc_value <= ({tx_crc_crc_next[0], tx_crc_crc_next[1], tx_crc_crc_next[2], tx_crc_crc_next[3], tx_crc_crc_next[4], tx_crc_crc_next[5], tx_crc_crc_next[6], tx_crc_crc_next[7], tx_crc_crc_next[8], tx_crc_crc_next[9], tx_crc_crc_next[10], tx_crc_crc_next[11], tx_crc_crc_next[12], tx_crc_crc_next[13], tx_crc_crc_next[14], tx_crc_crc_next[15], tx_crc_crc_next[16], tx_crc_crc_next[17], tx_crc_crc_next[18], tx_crc_crc_next[19], tx_crc_crc_next[20], tx_crc_crc_next[21], tx_crc_crc_next[22], tx_crc_crc_next[23], tx_crc_crc_next[24], tx_crc_crc_next[25], tx_crc_crc_next[26], tx_crc_crc_next[27], tx_crc_crc_next[28], tx_crc_crc_next[29], tx_crc_crc_next[30], tx_crc_crc_next[31]} ^ 32'd4294967295);
		tx_crc_error <= (tx_crc_crc_next != 32'd3338984827);
	end
// synthesis translate_off
	dummy_d_1 <= dummy_s;
// synthesis translate_on
end

// synthesis translate_off
reg dummy_d_2;
// synthesis translate_on
always @(*) begin
	tx_crc_crc_next <= 32'd0;
	tx_crc_crc_next[0] <= (((tx_crc_crc_prev[24] ^ tx_crc_crc_prev[30]) ^ tx_crc_data1[1]) ^ tx_crc_data1[7]);
	tx_crc_crc_next[1] <= (((((((tx_crc_crc_prev[25] ^ tx_crc_crc_prev[31]) ^ tx_crc_data1[0]) ^ tx_crc_data1[6]) ^ tx_crc_crc_prev[24]) ^ tx_crc_crc_prev[30]) ^ tx_crc_data1[1]) ^ tx_crc_data1[7]);
	tx_crc_crc_next[2] <= (((((((((tx_crc_crc_prev[26] ^ tx_crc_data1[5]) ^ tx_crc_crc_prev[25]) ^ tx_crc_crc_prev[31]) ^ tx_crc_data1[0]) ^ tx_crc_data1[6]) ^ tx_crc_crc_prev[24]) ^ tx_crc_crc_prev[30]) ^ tx_crc_data1[1]) ^ tx_crc_data1[7]);
	tx_crc_crc_next[3] <= (((((((tx_crc_crc_prev[27] ^ tx_crc_data1[4]) ^ tx_crc_crc_prev[26]) ^ tx_crc_data1[5]) ^ tx_crc_crc_prev[25]) ^ tx_crc_crc_prev[31]) ^ tx_crc_data1[0]) ^ tx_crc_data1[6]);
	tx_crc_crc_next[4] <= (((((((((tx_crc_crc_prev[28] ^ tx_crc_data1[3]) ^ tx_crc_crc_prev[27]) ^ tx_crc_data1[4]) ^ tx_crc_crc_prev[26]) ^ tx_crc_data1[5]) ^ tx_crc_crc_prev[24]) ^ tx_crc_crc_prev[30]) ^ tx_crc_data1[1]) ^ tx_crc_data1[7]);
	tx_crc_crc_next[5] <= (((((((((((((tx_crc_crc_prev[29] ^ tx_crc_data1[2]) ^ tx_crc_crc_prev[28]) ^ tx_crc_data1[3]) ^ tx_crc_crc_prev[27]) ^ tx_crc_data1[4]) ^ tx_crc_crc_prev[25]) ^ tx_crc_crc_prev[31]) ^ tx_crc_data1[0]) ^ tx_crc_data1[6]) ^ tx_crc_crc_prev[24]) ^ tx_crc_crc_prev[30]) ^ tx_crc_data1[1]) ^ tx_crc_data1[7]);
	tx_crc_crc_next[6] <= (((((((((((tx_crc_crc_prev[30] ^ tx_crc_data1[1]) ^ tx_crc_crc_prev[29]) ^ tx_crc_data1[2]) ^ tx_crc_crc_prev[28]) ^ tx_crc_data1[3]) ^ tx_crc_crc_prev[26]) ^ tx_crc_data1[5]) ^ tx_crc_crc_prev[25]) ^ tx_crc_crc_prev[31]) ^ tx_crc_data1[0]) ^ tx_crc_data1[6]);
	tx_crc_crc_next[7] <= (((((((((tx_crc_crc_prev[31] ^ tx_crc_data1[0]) ^ tx_crc_crc_prev[29]) ^ tx_crc_data1[2]) ^ tx_crc_crc_prev[27]) ^ tx_crc_data1[4]) ^ tx_crc_crc_prev[26]) ^ tx_crc_data1[5]) ^ tx_crc_crc_prev[24]) ^ tx_crc_data1[7]);
	tx_crc_crc_next[8] <= ((((((((tx_crc_crc_prev[0] ^ tx_crc_crc_prev[28]) ^ tx_crc_data1[3]) ^ tx_crc_crc_prev[27]) ^ tx_crc_data1[4]) ^ tx_crc_crc_prev[25]) ^ tx_crc_data1[6]) ^ tx_crc_crc_prev[24]) ^ tx_crc_data1[7]);
	tx_crc_crc_next[9] <= ((((((((tx_crc_crc_prev[1] ^ tx_crc_crc_prev[29]) ^ tx_crc_data1[2]) ^ tx_crc_crc_prev[28]) ^ tx_crc_data1[3]) ^ tx_crc_crc_prev[26]) ^ tx_crc_data1[5]) ^ tx_crc_crc_prev[25]) ^ tx_crc_data1[6]);
	tx_crc_crc_next[10] <= ((((((((tx_crc_crc_prev[2] ^ tx_crc_crc_prev[29]) ^ tx_crc_data1[2]) ^ tx_crc_crc_prev[27]) ^ tx_crc_data1[4]) ^ tx_crc_crc_prev[26]) ^ tx_crc_data1[5]) ^ tx_crc_crc_prev[24]) ^ tx_crc_data1[7]);
	tx_crc_crc_next[11] <= ((((((((tx_crc_crc_prev[3] ^ tx_crc_crc_prev[28]) ^ tx_crc_data1[3]) ^ tx_crc_crc_prev[27]) ^ tx_crc_data1[4]) ^ tx_crc_crc_prev[25]) ^ tx_crc_data1[6]) ^ tx_crc_crc_prev[24]) ^ tx_crc_data1[7]);
	tx_crc_crc_next[12] <= ((((((((((((tx_crc_crc_prev[4] ^ tx_crc_crc_prev[29]) ^ tx_crc_data1[2]) ^ tx_crc_crc_prev[28]) ^ tx_crc_data1[3]) ^ tx_crc_crc_prev[26]) ^ tx_crc_data1[5]) ^ tx_crc_crc_prev[25]) ^ tx_crc_data1[6]) ^ tx_crc_crc_prev[24]) ^ tx_crc_crc_prev[30]) ^ tx_crc_data1[1]) ^ tx_crc_data1[7]);
	tx_crc_crc_next[13] <= ((((((((((((tx_crc_crc_prev[5] ^ tx_crc_crc_prev[30]) ^ tx_crc_data1[1]) ^ tx_crc_crc_prev[29]) ^ tx_crc_data1[2]) ^ tx_crc_crc_prev[27]) ^ tx_crc_data1[4]) ^ tx_crc_crc_prev[26]) ^ tx_crc_data1[5]) ^ tx_crc_crc_prev[25]) ^ tx_crc_crc_prev[31]) ^ tx_crc_data1[0]) ^ tx_crc_data1[6]);
	tx_crc_crc_next[14] <= ((((((((((tx_crc_crc_prev[6] ^ tx_crc_crc_prev[31]) ^ tx_crc_data1[0]) ^ tx_crc_crc_prev[30]) ^ tx_crc_data1[1]) ^ tx_crc_crc_prev[28]) ^ tx_crc_data1[3]) ^ tx_crc_crc_prev[27]) ^ tx_crc_data1[4]) ^ tx_crc_crc_prev[26]) ^ tx_crc_data1[5]);
	tx_crc_crc_next[15] <= ((((((((tx_crc_crc_prev[7] ^ tx_crc_crc_prev[31]) ^ tx_crc_data1[0]) ^ tx_crc_crc_prev[29]) ^ tx_crc_data1[2]) ^ tx_crc_crc_prev[28]) ^ tx_crc_data1[3]) ^ tx_crc_crc_prev[27]) ^ tx_crc_data1[4]);
	tx_crc_crc_next[16] <= ((((((tx_crc_crc_prev[8] ^ tx_crc_crc_prev[29]) ^ tx_crc_data1[2]) ^ tx_crc_crc_prev[28]) ^ tx_crc_data1[3]) ^ tx_crc_crc_prev[24]) ^ tx_crc_data1[7]);
	tx_crc_crc_next[17] <= ((((((tx_crc_crc_prev[9] ^ tx_crc_crc_prev[30]) ^ tx_crc_data1[1]) ^ tx_crc_crc_prev[29]) ^ tx_crc_data1[2]) ^ tx_crc_crc_prev[25]) ^ tx_crc_data1[6]);
	tx_crc_crc_next[18] <= ((((((tx_crc_crc_prev[10] ^ tx_crc_crc_prev[31]) ^ tx_crc_data1[0]) ^ tx_crc_crc_prev[30]) ^ tx_crc_data1[1]) ^ tx_crc_crc_prev[26]) ^ tx_crc_data1[5]);
	tx_crc_crc_next[19] <= ((((tx_crc_crc_prev[11] ^ tx_crc_crc_prev[31]) ^ tx_crc_data1[0]) ^ tx_crc_crc_prev[27]) ^ tx_crc_data1[4]);
	tx_crc_crc_next[20] <= ((tx_crc_crc_prev[12] ^ tx_crc_crc_prev[28]) ^ tx_crc_data1[3]);
	tx_crc_crc_next[21] <= ((tx_crc_crc_prev[13] ^ tx_crc_crc_prev[29]) ^ tx_crc_data1[2]);
	tx_crc_crc_next[22] <= ((tx_crc_crc_prev[14] ^ tx_crc_crc_prev[24]) ^ tx_crc_data1[7]);
	tx_crc_crc_next[23] <= ((((((tx_crc_crc_prev[15] ^ tx_crc_crc_prev[25]) ^ tx_crc_data1[6]) ^ tx_crc_crc_prev[24]) ^ tx_crc_crc_prev[30]) ^ tx_crc_data1[1]) ^ tx_crc_data1[7]);
	tx_crc_crc_next[24] <= ((((((tx_crc_crc_prev[16] ^ tx_crc_crc_prev[26]) ^ tx_crc_data1[5]) ^ tx_crc_crc_prev[25]) ^ tx_crc_crc_prev[31]) ^ tx_crc_data1[0]) ^ tx_crc_data1[6]);
	tx_crc_crc_next[25] <= ((((tx_crc_crc_prev[17] ^ tx_crc_crc_prev[27]) ^ tx_crc_data1[4]) ^ tx_crc_crc_prev[26]) ^ tx_crc_data1[5]);
	tx_crc_crc_next[26] <= ((((((((tx_crc_crc_prev[18] ^ tx_crc_crc_prev[28]) ^ tx_crc_data1[3]) ^ tx_crc_crc_prev[27]) ^ tx_crc_data1[4]) ^ tx_crc_crc_prev[24]) ^ tx_crc_crc_prev[30]) ^ tx_crc_data1[1]) ^ tx_crc_data1[7]);
	tx_crc_crc_next[27] <= ((((((((tx_crc_crc_prev[19] ^ tx_crc_crc_prev[29]) ^ tx_crc_data1[2]) ^ tx_crc_crc_prev[28]) ^ tx_crc_data1[3]) ^ tx_crc_crc_prev[25]) ^ tx_crc_crc_prev[31]) ^ tx_crc_data1[0]) ^ tx_crc_data1[6]);
	tx_crc_crc_next[28] <= ((((((tx_crc_crc_prev[20] ^ tx_crc_crc_prev[30]) ^ tx_crc_data1[1]) ^ tx_crc_crc_prev[29]) ^ tx_crc_data1[2]) ^ tx_crc_crc_prev[26]) ^ tx_crc_data1[5]);
	tx_crc_crc_next[29] <= ((((((tx_crc_crc_prev[21] ^ tx_crc_crc_prev[31]) ^ tx_crc_data1[0]) ^ tx_crc_crc_prev[30]) ^ tx_crc_data1[1]) ^ tx_crc_crc_prev[27]) ^ tx_crc_data1[4]);
	tx_crc_crc_next[30] <= ((((tx_crc_crc_prev[22] ^ tx_crc_crc_prev[31]) ^ tx_crc_data1[0]) ^ tx_crc_crc_prev[28]) ^ tx_crc_data1[3]);
	tx_crc_crc_next[31] <= ((tx_crc_crc_prev[23] ^ tx_crc_crc_prev[29]) ^ tx_crc_data1[2]);
// synthesis translate_off
	dummy_d_2 <= dummy_s;
// synthesis translate_on
end

// synthesis translate_off
reg dummy_d_3;
// synthesis translate_on
always @(*) begin
	tx_crc_sink_ready <= 1'd0;
	tx_crc_source_valid <= 1'd0;
	tx_crc_source_first <= 1'd0;
	tx_crc_source_last <= 1'd0;
	tx_crc_source_payload_data <= 8'd0;
	tx_crc_source_payload_be <= 1'd0;
	tx_crc_source_payload_error <= 1'd0;
	tx_crc_ce <= 1'd0;
	tx_crc_reset <= 1'd0;
	tx_crc_is_ongoing0 <= 1'd0;
	tx_crc_is_ongoing1 <= 1'd0;
	txdatapath_bufferizeendpoints_next_state <= 2'd0;
	tx_crc_crc_packet_clockdomainsrenamer1_next_value0 <= 32'd0;
	tx_crc_crc_packet_clockdomainsrenamer1_next_value_ce0 <= 1'd0;
	tx_crc_be0_clockdomainsrenamer1_next_value1 <= 1'd0;
	tx_crc_be0_clockdomainsrenamer1_next_value_ce1 <= 1'd0;
	txdatapath_bufferizeendpoints_next_state <= txdatapath_bufferizeendpoints_state;
	case (txdatapath_bufferizeendpoints_state)
		1'd1: begin
			tx_crc_ce <= (tx_crc_sink_valid & tx_crc_source_ready);
			tx_crc_source_valid <= tx_crc_sink_valid;
			tx_crc_sink_ready <= tx_crc_source_ready;
			tx_crc_source_first <= tx_crc_sink_first;
			tx_crc_source_last <= tx_crc_sink_last;
			tx_crc_source_payload_data <= tx_crc_sink_payload_data;
			tx_crc_source_payload_be <= tx_crc_sink_payload_be;
			tx_crc_source_payload_error <= tx_crc_sink_payload_error;
			tx_crc_source_last <= 1'd0;
			tx_crc_source_payload_be <= 1'd1;
			if (tx_crc_sink_last) begin
				if (tx_crc_sink_payload_be) begin
					tx_crc_source_payload_data <= slice_proxy[7:0];
				end
				if ((1'd0 & (tx_crc_sink_payload_be <= 4'd15))) begin
					tx_crc_source_last <= 1'd1;
					tx_crc_source_payload_be <= ((tx_crc_sink_payload_be <<< 3'd4) | 4'd15);
				end
			end
			if (((tx_crc_sink_valid & tx_crc_sink_last) & tx_crc_source_ready)) begin
				if ((1'd0 & (tx_crc_sink_payload_be <= 4'd15))) begin
					txdatapath_bufferizeendpoints_next_state <= 1'd0;
				end else begin
					tx_crc_crc_packet_clockdomainsrenamer1_next_value0 <= tx_crc_value;
					tx_crc_crc_packet_clockdomainsrenamer1_next_value_ce0 <= 1'd1;
					if (1'd0) begin
						tx_crc_be0_clockdomainsrenamer1_next_value1 <= (tx_crc_sink_payload_be >>> 3'd4);
						tx_crc_be0_clockdomainsrenamer1_next_value_ce1 <= 1'd1;
					end else begin
						tx_crc_be0_clockdomainsrenamer1_next_value1 <= tx_crc_sink_payload_be;
						tx_crc_be0_clockdomainsrenamer1_next_value_ce1 <= 1'd1;
					end
					txdatapath_bufferizeendpoints_next_state <= 2'd2;
				end
			end
		end
		2'd2: begin
			tx_crc_source_valid <= 1'd1;
			tx_crc_source_payload_be <= 1'd1;
			case (tx_crc_cnt)
				1'd0: begin
					tx_crc_source_payload_data <= tx_crc_crc_packet[31:24];
				end
				1'd1: begin
					tx_crc_source_payload_data <= tx_crc_crc_packet[23:16];
				end
				2'd2: begin
					tx_crc_source_payload_data <= tx_crc_crc_packet[15:8];
				end
				default: begin
					tx_crc_source_payload_data <= tx_crc_crc_packet[7:0];
				end
			endcase
			if (tx_crc_cnt_done) begin
				tx_crc_source_last <= 1'd1;
				if (tx_crc_source_ready) begin
					txdatapath_bufferizeendpoints_next_state <= 1'd0;
				end
			end
			tx_crc_is_ongoing1 <= 1'd1;
		end
		default: begin
			tx_crc_reset <= 1'd1;
			tx_crc_sink_ready <= 1'd1;
			if (tx_crc_sink_valid) begin
				tx_crc_sink_ready <= 1'd0;
				txdatapath_bufferizeendpoints_next_state <= 1'd1;
			end
			tx_crc_is_ongoing0 <= 1'd1;
		end
	endcase
// synthesis translate_off
	dummy_d_3 <= dummy_s;
// synthesis translate_on
end
assign tx_crc_pipe_valid_sink_ready = ((~tx_crc_pipe_valid_source_valid) | tx_crc_pipe_valid_source_ready);
assign tx_crc_pipe_valid_sink_valid = tx_crc_sink_sink_valid;
assign tx_crc_sink_sink_ready = tx_crc_pipe_valid_sink_ready;
assign tx_crc_pipe_valid_sink_first = tx_crc_sink_sink_first;
assign tx_crc_pipe_valid_sink_last = tx_crc_sink_sink_last;
assign tx_crc_pipe_valid_sink_payload_data = tx_crc_sink_sink_payload_data;
assign tx_crc_pipe_valid_sink_payload_be = tx_crc_sink_sink_payload_be;
assign tx_crc_pipe_valid_sink_payload_error = tx_crc_sink_sink_payload_error;
assign tx_crc_source_source_valid = tx_crc_pipe_valid_source_valid;
assign tx_crc_pipe_valid_source_ready = tx_crc_source_source_ready;
assign tx_crc_source_source_first = tx_crc_pipe_valid_source_first;
assign tx_crc_source_source_last = tx_crc_pipe_valid_source_last;
assign tx_crc_source_source_payload_data = tx_crc_pipe_valid_source_payload_data;
assign tx_crc_source_source_payload_be = tx_crc_pipe_valid_source_payload_be;
assign tx_crc_source_source_payload_error = tx_crc_pipe_valid_source_payload_error;

// synthesis translate_off
reg dummy_d_4;
// synthesis translate_on
always @(*) begin
	tx_preamble_sink_ready <= 1'd0;
	tx_preamble_source_valid <= 1'd0;
	tx_preamble_source_first <= 1'd0;
	tx_preamble_source_last <= 1'd0;
	tx_preamble_source_payload_data <= 8'd0;
	tx_preamble_source_payload_be <= 1'd0;
	tx_preamble_source_payload_error <= 1'd0;
	txdatapath_liteethmacpreambleinserter_next_state <= 2'd0;
	tx_preamble_count_clockdomainsrenamer2_next_value <= 3'd0;
	tx_preamble_count_clockdomainsrenamer2_next_value_ce <= 1'd0;
	tx_preamble_source_payload_data <= tx_preamble_sink_payload_data;
	tx_preamble_source_payload_be <= tx_preamble_sink_payload_be;
	txdatapath_liteethmacpreambleinserter_next_state <= txdatapath_liteethmacpreambleinserter_state;
	case (txdatapath_liteethmacpreambleinserter_state)
		1'd1: begin
			tx_preamble_source_valid <= 1'd1;
			tx_preamble_source_payload_be <= 1'd1;
			case (tx_preamble_count)
				1'd0: begin
					tx_preamble_source_payload_data <= tx_preamble_preamble[7:0];
				end
				1'd1: begin
					tx_preamble_source_payload_data <= tx_preamble_preamble[15:8];
				end
				2'd2: begin
					tx_preamble_source_payload_data <= tx_preamble_preamble[23:16];
				end
				2'd3: begin
					tx_preamble_source_payload_data <= tx_preamble_preamble[31:24];
				end
				3'd4: begin
					tx_preamble_source_payload_data <= tx_preamble_preamble[39:32];
				end
				3'd5: begin
					tx_preamble_source_payload_data <= tx_preamble_preamble[47:40];
				end
				3'd6: begin
					tx_preamble_source_payload_data <= tx_preamble_preamble[55:48];
				end
				default: begin
					tx_preamble_source_payload_data <= tx_preamble_preamble[63:56];
				end
			endcase
			if (tx_preamble_source_ready) begin
				if ((tx_preamble_count == 3'd7)) begin
					txdatapath_liteethmacpreambleinserter_next_state <= 2'd2;
				end else begin
					tx_preamble_count_clockdomainsrenamer2_next_value <= (tx_preamble_count + 1'd1);
					tx_preamble_count_clockdomainsrenamer2_next_value_ce <= 1'd1;
				end
			end
		end
		2'd2: begin
			tx_preamble_source_valid <= tx_preamble_sink_valid;
			tx_preamble_sink_ready <= tx_preamble_source_ready;
			tx_preamble_source_first <= tx_preamble_sink_first;
			tx_preamble_source_last <= tx_preamble_sink_last;
			tx_preamble_source_payload_error <= tx_preamble_sink_payload_error;
			if (((tx_preamble_sink_valid & tx_preamble_sink_last) & tx_preamble_source_ready)) begin
				txdatapath_liteethmacpreambleinserter_next_state <= 1'd0;
			end
		end
		default: begin
			tx_preamble_sink_ready <= 1'd1;
			tx_preamble_count_clockdomainsrenamer2_next_value <= 1'd0;
			tx_preamble_count_clockdomainsrenamer2_next_value_ce <= 1'd1;
			if (tx_preamble_sink_valid) begin
				tx_preamble_sink_ready <= 1'd0;
				txdatapath_liteethmacpreambleinserter_next_state <= 1'd1;
			end
		end
	endcase
// synthesis translate_off
	dummy_d_4 <= dummy_s;
// synthesis translate_on
end
assign txdatapath_write_rst = (sys_rst | txdatapath_write_ext_rst);
assign txdatapath_cdc_reset_write = txdatapath_write_ext_rst;
assign txdatapath_cdc_reset_read = txdatapath_ext_rst_to;
assign txdatapath_cdc_sink_valid = txdatapath_sink_sink_valid;
assign txdatapath_sink_sink_ready = txdatapath_cdc_sink_ready;
assign txdatapath_cdc_sink_first = txdatapath_sink_sink_first;
assign txdatapath_cdc_sink_last = txdatapath_sink_sink_last;
assign txdatapath_cdc_sink_payload_data = txdatapath_sink_sink_payload_data;
assign txdatapath_cdc_sink_payload_be = txdatapath_sink_sink_payload_be;
assign txdatapath_cdc_sink_payload_error = txdatapath_sink_sink_payload_error;
assign txdatapath_cdc_source_ready = txdatapath_source_source_ready;
assign txdatapath_source_source_first = txdatapath_cdc_source_first;
assign txdatapath_source_source_last = txdatapath_cdc_source_last;
assign txdatapath_source_source_payload_data = txdatapath_cdc_source_payload_data;
assign txdatapath_source_source_payload_be = txdatapath_cdc_source_payload_be;
assign txdatapath_source_source_payload_error = txdatapath_cdc_source_payload_error;
assign txdatapath_source_source_valid = ((txdatapath_cdc_source_valid & (~txdatapath_ext_rst_to)) & (~eth_tx_rst));
assign txdatapath_cdc_asyncfifo_din = {txdatapath_cdc_fifo_in_last, txdatapath_cdc_fifo_in_first, txdatapath_cdc_fifo_in_payload_error, txdatapath_cdc_fifo_in_payload_be, txdatapath_cdc_fifo_in_payload_data};
assign {txdatapath_cdc_fifo_out_last, txdatapath_cdc_fifo_out_first, txdatapath_cdc_fifo_out_payload_error, txdatapath_cdc_fifo_out_payload_be, txdatapath_cdc_fifo_out_payload_data} = txdatapath_cdc_asyncfifo_dout;
assign txdatapath_cdc_sink_ready = txdatapath_cdc_asyncfifo_writable;
assign txdatapath_cdc_asyncfifo_we = txdatapath_cdc_sink_valid;
assign txdatapath_cdc_fifo_in_first = txdatapath_cdc_sink_first;
assign txdatapath_cdc_fifo_in_last = txdatapath_cdc_sink_last;
assign txdatapath_cdc_fifo_in_payload_data = txdatapath_cdc_sink_payload_data;
assign txdatapath_cdc_fifo_in_payload_be = txdatapath_cdc_sink_payload_be;
assign txdatapath_cdc_fifo_in_payload_error = txdatapath_cdc_sink_payload_error;
assign txdatapath_cdc_source_valid = txdatapath_cdc_asyncfifo_readable;
assign txdatapath_cdc_source_first = txdatapath_cdc_fifo_out_first;
assign txdatapath_cdc_source_last = txdatapath_cdc_fifo_out_last;
assign txdatapath_cdc_source_payload_data = txdatapath_cdc_fifo_out_payload_data;
assign txdatapath_cdc_source_payload_be = txdatapath_cdc_fifo_out_payload_be;
assign txdatapath_cdc_source_payload_error = txdatapath_cdc_fifo_out_payload_error;
assign txdatapath_cdc_asyncfifo_re = txdatapath_cdc_source_ready;
assign txdatapath_cdc_graycounter0_ce = (txdatapath_cdc_asyncfifo_writable & txdatapath_cdc_asyncfifo_we);
assign txdatapath_cdc_graycounter1_ce = (txdatapath_cdc_asyncfifo_readable & txdatapath_cdc_asyncfifo_re);
assign txdatapath_cdc_asyncfifo_writable = (((txdatapath_cdc_graycounter0_q[5] == txdatapath_cdc_consume_wdomain[5]) | (txdatapath_cdc_graycounter0_q[4] == txdatapath_cdc_consume_wdomain[4])) | (txdatapath_cdc_graycounter0_q[3:0] != txdatapath_cdc_consume_wdomain[3:0]));
assign txdatapath_cdc_asyncfifo_readable = (txdatapath_cdc_graycounter1_q != txdatapath_cdc_produce_rdomain);
assign txdatapath_cdc_wrport_adr = txdatapath_cdc_graycounter0_q_binary[4:0];
assign txdatapath_cdc_wrport_dat_w = txdatapath_cdc_asyncfifo_din;
assign txdatapath_cdc_wrport_we = txdatapath_cdc_graycounter0_ce;
assign txdatapath_cdc_rdport_adr = txdatapath_cdc_graycounter1_q_next_binary[4:0];
assign txdatapath_cdc_asyncfifo_dout = txdatapath_cdc_rdport_dat_r;

// synthesis translate_off
reg dummy_d_5;
// synthesis translate_on
always @(*) begin
	txdatapath_cdc_graycounter0_q_next_binary <= 6'd0;
	if (txdatapath_cdc_graycounter0_ce) begin
		txdatapath_cdc_graycounter0_q_next_binary <= (txdatapath_cdc_graycounter0_q_binary + 1'd1);
	end else begin
		txdatapath_cdc_graycounter0_q_next_binary <= txdatapath_cdc_graycounter0_q_binary;
	end
// synthesis translate_off
	dummy_d_5 <= dummy_s;
// synthesis translate_on
end
assign txdatapath_cdc_graycounter0_q_next = (txdatapath_cdc_graycounter0_q_next_binary ^ txdatapath_cdc_graycounter0_q_next_binary[5:1]);

// synthesis translate_off
reg dummy_d_6;
// synthesis translate_on
always @(*) begin
	txdatapath_cdc_graycounter1_q_next_binary <= 6'd0;
	if (txdatapath_cdc_graycounter1_ce) begin
		txdatapath_cdc_graycounter1_q_next_binary <= (txdatapath_cdc_graycounter1_q_binary + 1'd1);
	end else begin
		txdatapath_cdc_graycounter1_q_next_binary <= txdatapath_cdc_graycounter1_q_binary;
	end
// synthesis translate_off
	dummy_d_6 <= dummy_s;
// synthesis translate_on
end
assign txdatapath_cdc_graycounter1_q_next = (txdatapath_cdc_graycounter1_q_next_binary ^ txdatapath_cdc_graycounter1_q_next_binary[5:1]);
assign tx_packet_fifo_payload_fifo_sink_first = tx_packet_fifo_sink_sink_first;
assign tx_packet_fifo_payload_fifo_sink_last = tx_packet_fifo_sink_sink_last;
assign tx_packet_fifo_payload_fifo_sink_payload_data = tx_packet_fifo_sink_sink_payload_data;
assign tx_packet_fifo_payload_fifo_sink_payload_be = tx_packet_fifo_sink_sink_payload_be;
assign tx_packet_fifo_payload_fifo_sink_payload_error = tx_packet_fifo_sink_sink_payload_error;
assign tx_packet_fifo_param_fifo_sink_valid = ((tx_packet_fifo_sink_sink_valid & tx_packet_fifo_sink_sink_last) & tx_packet_fifo_payload_fifo_sink_ready);
assign tx_packet_fifo_payload_fifo_sink_valid = (tx_packet_fifo_sink_sink_valid & tx_packet_fifo_param_fifo_sink_ready);
assign tx_packet_fifo_sink_sink_ready = (tx_packet_fifo_param_fifo_sink_ready & tx_packet_fifo_payload_fifo_sink_ready);
assign tx_packet_fifo_source_source_valid = tx_packet_fifo_param_fifo_source_valid;

// synthesis translate_off
reg dummy_d_7;
// synthesis translate_on
always @(*) begin
	tx_packet_fifo_source_source_first <= 1'd0;
	tx_packet_fifo_source_source_first <= tx_packet_fifo_param_fifo_source_first;
	tx_packet_fifo_source_source_first <= tx_packet_fifo_payload_fifo_source_first;
// synthesis translate_off
	dummy_d_7 <= dummy_s;
// synthesis translate_on
end
assign tx_packet_fifo_source_source_last = tx_packet_fifo_payload_fifo_source_last;
assign tx_packet_fifo_source_source_payload_data = tx_packet_fifo_payload_fifo_source_payload_data;
assign tx_packet_fifo_source_source_payload_be = tx_packet_fifo_payload_fifo_source_payload_be;
assign tx_packet_fifo_source_source_payload_error = tx_packet_fifo_payload_fifo_source_payload_error;
assign tx_packet_fifo_param_fifo_source_ready = ((tx_packet_fifo_source_source_valid & tx_packet_fifo_source_source_last) & tx_packet_fifo_source_source_ready);
assign tx_packet_fifo_payload_fifo_source_ready = (tx_packet_fifo_source_source_valid & tx_packet_fifo_source_source_ready);
assign tx_packet_fifo_payload_fifo_syncfifo_din = {tx_packet_fifo_payload_fifo_fifo_in_last, tx_packet_fifo_payload_fifo_fifo_in_first, tx_packet_fifo_payload_fifo_fifo_in_payload_error, tx_packet_fifo_payload_fifo_fifo_in_payload_be, tx_packet_fifo_payload_fifo_fifo_in_payload_data};
assign {tx_packet_fifo_payload_fifo_fifo_out_last, tx_packet_fifo_payload_fifo_fifo_out_first, tx_packet_fifo_payload_fifo_fifo_out_payload_error, tx_packet_fifo_payload_fifo_fifo_out_payload_be, tx_packet_fifo_payload_fifo_fifo_out_payload_data} = tx_packet_fifo_payload_fifo_syncfifo_dout;
assign tx_packet_fifo_payload_fifo_sink_ready = tx_packet_fifo_payload_fifo_syncfifo_writable;
assign tx_packet_fifo_payload_fifo_syncfifo_we = tx_packet_fifo_payload_fifo_sink_valid;
assign tx_packet_fifo_payload_fifo_fifo_in_first = tx_packet_fifo_payload_fifo_sink_first;
assign tx_packet_fifo_payload_fifo_fifo_in_last = tx_packet_fifo_payload_fifo_sink_last;
assign tx_packet_fifo_payload_fifo_fifo_in_payload_data = tx_packet_fifo_payload_fifo_sink_payload_data;
assign tx_packet_fifo_payload_fifo_fifo_in_payload_be = tx_packet_fifo_payload_fifo_sink_payload_be;
assign tx_packet_fifo_payload_fifo_fifo_in_payload_error = tx_packet_fifo_payload_fifo_sink_payload_error;
assign tx_packet_fifo_payload_fifo_source_valid = tx_packet_fifo_payload_fifo_readable;
assign tx_packet_fifo_payload_fifo_source_first = tx_packet_fifo_payload_fifo_fifo_out_first;
assign tx_packet_fifo_payload_fifo_source_last = tx_packet_fifo_payload_fifo_fifo_out_last;
assign tx_packet_fifo_payload_fifo_source_payload_data = tx_packet_fifo_payload_fifo_fifo_out_payload_data;
assign tx_packet_fifo_payload_fifo_source_payload_be = tx_packet_fifo_payload_fifo_fifo_out_payload_be;
assign tx_packet_fifo_payload_fifo_source_payload_error = tx_packet_fifo_payload_fifo_fifo_out_payload_error;
assign tx_packet_fifo_payload_fifo_re = tx_packet_fifo_payload_fifo_source_ready;
assign tx_packet_fifo_payload_fifo_syncfifo_re = (tx_packet_fifo_payload_fifo_syncfifo_readable & ((~tx_packet_fifo_payload_fifo_readable) | tx_packet_fifo_payload_fifo_re));
assign tx_packet_fifo_payload_fifo_level1 = (tx_packet_fifo_payload_fifo_level0 + tx_packet_fifo_payload_fifo_readable);

// synthesis translate_off
reg dummy_d_8;
// synthesis translate_on
always @(*) begin
	tx_packet_fifo_payload_fifo_wrport_adr <= 11'd0;
	if (tx_packet_fifo_payload_fifo_replace) begin
		tx_packet_fifo_payload_fifo_wrport_adr <= (tx_packet_fifo_payload_fifo_produce - 1'd1);
	end else begin
		tx_packet_fifo_payload_fifo_wrport_adr <= tx_packet_fifo_payload_fifo_produce;
	end
// synthesis translate_off
	dummy_d_8 <= dummy_s;
// synthesis translate_on
end
assign tx_packet_fifo_payload_fifo_wrport_dat_w = tx_packet_fifo_payload_fifo_syncfifo_din;
assign tx_packet_fifo_payload_fifo_wrport_we = (tx_packet_fifo_payload_fifo_syncfifo_we & (tx_packet_fifo_payload_fifo_syncfifo_writable | tx_packet_fifo_payload_fifo_replace));
assign tx_packet_fifo_payload_fifo_do_read = (tx_packet_fifo_payload_fifo_syncfifo_readable & tx_packet_fifo_payload_fifo_syncfifo_re);
assign tx_packet_fifo_payload_fifo_rdport_adr = tx_packet_fifo_payload_fifo_consume;
assign tx_packet_fifo_payload_fifo_syncfifo_dout = tx_packet_fifo_payload_fifo_rdport_dat_r;
assign tx_packet_fifo_payload_fifo_rdport_re = tx_packet_fifo_payload_fifo_do_read;
assign tx_packet_fifo_payload_fifo_syncfifo_writable = (tx_packet_fifo_payload_fifo_level0 != 12'd2048);
assign tx_packet_fifo_payload_fifo_syncfifo_readable = (tx_packet_fifo_payload_fifo_level0 != 1'd0);
assign tx_packet_fifo_param_fifo_syncfifo_din = {tx_packet_fifo_param_fifo_fifo_in_last, tx_packet_fifo_param_fifo_fifo_in_first, tx_packet_fifo_param_fifo_fifo_in_param_dummy};
assign {tx_packet_fifo_param_fifo_fifo_out_last, tx_packet_fifo_param_fifo_fifo_out_first, tx_packet_fifo_param_fifo_fifo_out_param_dummy} = tx_packet_fifo_param_fifo_syncfifo_dout;
assign tx_packet_fifo_param_fifo_sink_ready = tx_packet_fifo_param_fifo_syncfifo_writable;
assign tx_packet_fifo_param_fifo_syncfifo_we = tx_packet_fifo_param_fifo_sink_valid;
assign tx_packet_fifo_param_fifo_fifo_in_first = tx_packet_fifo_param_fifo_sink_first;
assign tx_packet_fifo_param_fifo_fifo_in_last = tx_packet_fifo_param_fifo_sink_last;
assign tx_packet_fifo_param_fifo_fifo_in_param_dummy = tx_packet_fifo_param_fifo_sink_param_dummy;
assign tx_packet_fifo_param_fifo_source_valid = tx_packet_fifo_param_fifo_readable;
assign tx_packet_fifo_param_fifo_source_first = tx_packet_fifo_param_fifo_fifo_out_first;
assign tx_packet_fifo_param_fifo_source_last = tx_packet_fifo_param_fifo_fifo_out_last;
assign tx_packet_fifo_param_fifo_source_param_dummy = tx_packet_fifo_param_fifo_fifo_out_param_dummy;
assign tx_packet_fifo_param_fifo_re = tx_packet_fifo_param_fifo_source_ready;
assign tx_packet_fifo_param_fifo_syncfifo_re = (tx_packet_fifo_param_fifo_syncfifo_readable & ((~tx_packet_fifo_param_fifo_readable) | tx_packet_fifo_param_fifo_re));
assign tx_packet_fifo_param_fifo_level1 = (tx_packet_fifo_param_fifo_level0 + tx_packet_fifo_param_fifo_readable);

// synthesis translate_off
reg dummy_d_9;
// synthesis translate_on
always @(*) begin
	tx_packet_fifo_param_fifo_wrport_adr <= 3'd0;
	if (tx_packet_fifo_param_fifo_replace) begin
		tx_packet_fifo_param_fifo_wrport_adr <= (tx_packet_fifo_param_fifo_produce - 1'd1);
	end else begin
		tx_packet_fifo_param_fifo_wrport_adr <= tx_packet_fifo_param_fifo_produce;
	end
// synthesis translate_off
	dummy_d_9 <= dummy_s;
// synthesis translate_on
end
assign tx_packet_fifo_param_fifo_wrport_dat_w = tx_packet_fifo_param_fifo_syncfifo_din;
assign tx_packet_fifo_param_fifo_wrport_we = (tx_packet_fifo_param_fifo_syncfifo_we & (tx_packet_fifo_param_fifo_syncfifo_writable | tx_packet_fifo_param_fifo_replace));
assign tx_packet_fifo_param_fifo_do_read = (tx_packet_fifo_param_fifo_syncfifo_readable & tx_packet_fifo_param_fifo_syncfifo_re);
assign tx_packet_fifo_param_fifo_rdport_adr = tx_packet_fifo_param_fifo_consume;
assign tx_packet_fifo_param_fifo_syncfifo_dout = tx_packet_fifo_param_fifo_rdport_dat_r;
assign tx_packet_fifo_param_fifo_rdport_re = tx_packet_fifo_param_fifo_do_read;
assign tx_packet_fifo_param_fifo_syncfifo_writable = (tx_packet_fifo_param_fifo_level0 != 3'd5);
assign tx_packet_fifo_param_fifo_syncfifo_readable = (tx_packet_fifo_param_fifo_level0 != 1'd0);

// synthesis translate_off
reg dummy_d_10;
// synthesis translate_on
always @(*) begin
	tx_gap_sink_ready <= 1'd0;
	tx_gap_source_valid <= 1'd0;
	tx_gap_source_first <= 1'd0;
	tx_gap_source_last <= 1'd0;
	tx_gap_source_payload_data <= 8'd0;
	tx_gap_source_payload_be <= 1'd0;
	tx_gap_source_payload_error <= 1'd0;
	txdatapath_resetinserter_next_state <= 1'd0;
	tx_gap_counter_clockdomainsrenamer3_next_value <= 4'd0;
	tx_gap_counter_clockdomainsrenamer3_next_value_ce <= 1'd0;
	txdatapath_resetinserter_next_state <= txdatapath_resetinserter_state;
	case (txdatapath_resetinserter_state)
		1'd1: begin
			tx_gap_counter_clockdomainsrenamer3_next_value <= (tx_gap_counter - 1'd1);
			tx_gap_counter_clockdomainsrenamer3_next_value_ce <= 1'd1;
			if ((tx_gap_counter == 1'd1)) begin
				txdatapath_resetinserter_next_state <= 1'd0;
			end
		end
		default: begin
			tx_gap_source_valid <= tx_gap_sink_valid;
			tx_gap_sink_ready <= tx_gap_source_ready;
			tx_gap_source_first <= tx_gap_sink_first;
			tx_gap_source_last <= tx_gap_sink_last;
			tx_gap_source_payload_data <= tx_gap_sink_payload_data;
			tx_gap_source_payload_be <= tx_gap_sink_payload_be;
			tx_gap_source_payload_error <= tx_gap_sink_payload_error;
			if (((tx_gap_sink_valid & tx_gap_sink_last) & tx_gap_sink_ready)) begin
				tx_gap_counter_clockdomainsrenamer3_next_value <= 4'd12;
				tx_gap_counter_clockdomainsrenamer3_next_value_ce <= 1'd1;
				txdatapath_resetinserter_next_state <= 1'd1;
			end
		end
	endcase
// synthesis translate_off
	dummy_d_10 <= dummy_s;
// synthesis translate_on
end
assign tx_padding_sink_valid = sink_valid;
assign sink_ready = tx_padding_sink_ready;
assign tx_padding_sink_first = sink_first;
assign tx_padding_sink_last = sink_last;
assign tx_padding_sink_payload_data = sink_payload_data;
assign tx_padding_sink_payload_be = sink_payload_be;
assign tx_padding_sink_payload_error = sink_payload_error;
assign tx_crc_sink_sink_valid = tx_padding_source_valid;
assign tx_padding_source_ready = tx_crc_sink_sink_ready;
assign tx_crc_sink_sink_first = tx_padding_source_first;
assign tx_crc_sink_sink_last = tx_padding_source_last;
assign tx_crc_sink_sink_payload_data = tx_padding_source_payload_data;
assign tx_crc_sink_sink_payload_be = tx_padding_source_payload_be;
assign tx_crc_sink_sink_payload_error = tx_padding_source_payload_error;
assign tx_preamble_sink_valid = tx_crc_source_valid;
assign tx_crc_source_ready = tx_preamble_sink_ready;
assign tx_preamble_sink_first = tx_crc_source_first;
assign tx_preamble_sink_last = tx_crc_source_last;
assign tx_preamble_sink_payload_data = tx_crc_source_payload_data;
assign tx_preamble_sink_payload_be = tx_crc_source_payload_be;
assign tx_preamble_sink_payload_error = tx_crc_source_payload_error;
assign txdatapath_sink_sink_valid = tx_preamble_source_valid;
assign tx_preamble_source_ready = txdatapath_sink_sink_ready;
assign txdatapath_sink_sink_first = tx_preamble_source_first;
assign txdatapath_sink_sink_last = tx_preamble_source_last;
assign txdatapath_sink_sink_payload_data = tx_preamble_source_payload_data;
assign txdatapath_sink_sink_payload_be = tx_preamble_source_payload_be;
assign txdatapath_sink_sink_payload_error = tx_preamble_source_payload_error;
assign tx_packet_fifo_sink_sink_valid = txdatapath_source_source_valid;
assign txdatapath_source_source_ready = tx_packet_fifo_sink_sink_ready;
assign tx_packet_fifo_sink_sink_first = txdatapath_source_source_first;
assign tx_packet_fifo_sink_sink_last = txdatapath_source_source_last;
assign tx_packet_fifo_sink_sink_payload_data = txdatapath_source_source_payload_data;
assign tx_packet_fifo_sink_sink_payload_be = txdatapath_source_source_payload_be;
assign tx_packet_fifo_sink_sink_payload_error = txdatapath_source_source_payload_error;
assign tx_gap_sink_valid = tx_packet_fifo_source_source_valid;
assign tx_packet_fifo_source_source_ready = tx_gap_sink_ready;
assign tx_gap_sink_first = tx_packet_fifo_source_source_first;
assign tx_gap_sink_last = tx_packet_fifo_source_source_last;
assign tx_gap_sink_payload_data = tx_packet_fifo_source_source_payload_data;
assign tx_gap_sink_payload_be = tx_packet_fifo_source_source_payload_be;
assign tx_gap_sink_payload_error = tx_packet_fifo_source_source_payload_error;
assign rmiitxshim_sink_valid = tx_gap_source_valid;
assign tx_gap_source_ready = rmiitxshim_sink_ready;
assign rmiitxshim_sink_first = tx_gap_source_first;
assign rmiitxshim_sink_last = tx_gap_source_last;
assign rmiitxshim_sink_payload_data = tx_gap_source_payload_data;
assign rmiitxshim_sink_payload_be = tx_gap_source_payload_be;
assign rmiitxshim_sink_payload_error = tx_gap_source_payload_error;
assign i = rx_packet_fifo_drop;
assign pulsesynchronizer0_i = rx_preamble_error;
assign pulsesynchronizer1_i = liteethmaccrc32checker_error;
assign rx_packet_fifo_used = (rx_packet_fifo_wr_ptr - (rx_packet_fifo_rd_ptr - rx_packet_fifo_out_valid));
assign rx_packet_fifo_full = (rx_packet_fifo_used == 12'd2048);
assign rx_packet_fifo_sink_sink_ready = 1'd1;
assign rx_packet_fifo_wr_port_adr = rx_packet_fifo_wr_ptr[10:0];
assign rx_packet_fifo_wr_port_dat_w = {rx_packet_fifo_sink_sink_last, rx_packet_fifo_sink_sink_first, {rx_packet_fifo_sink_sink_payload_error, rx_packet_fifo_sink_sink_payload_be, rx_packet_fifo_sink_sink_payload_data}};
assign rx_packet_fifo_wr_port_we = ((rx_packet_fifo_sink_sink_valid & (~rx_packet_fifo_overflow)) & (~rx_packet_fifo_full));
assign rx_packet_fifo_commit = ((((((rx_packet_fifo_sink_sink_valid & rx_packet_fifo_sink_sink_last) & (~rx_packet_fifo_overflow)) & (~rx_packet_fifo_full)) & (~rx_packet_fifo_invalid)) & (~rx_packet_fifo_discard)) & rx_packet_fifo_param_fifo_sink_ready);
assign rx_packet_fifo_param_fifo_sink_valid = rx_packet_fifo_commit;
assign rx_packet_fifo_param_fifo_sink_last = 1'd1;
assign rx_packet_fifo_drop = ((rx_packet_fifo_sink_sink_valid & rx_packet_fifo_sink_sink_last) & (~rx_packet_fifo_commit));
assign rx_packet_fifo_fetch = ((rx_packet_fifo_rd_ptr != rx_packet_fifo_commit_ptr) & ((~rx_packet_fifo_out_valid) | (rx_packet_fifo_source_source_valid & rx_packet_fifo_source_source_ready)));
assign rx_packet_fifo_rd_port_adr = rx_packet_fifo_rd_ptr[10:0];
assign rx_packet_fifo_rd_port_re = rx_packet_fifo_fetch;
assign {rx_packet_fifo_source_source_payload_error, rx_packet_fifo_source_source_payload_be, rx_packet_fifo_source_source_payload_data} = rx_packet_fifo_rd_port_dat_r[9:0];
assign rx_packet_fifo_source_source_first = rx_packet_fifo_rd_port_dat_r[10];
assign rx_packet_fifo_source_source_last = rx_packet_fifo_rd_port_dat_r[11];
assign rx_packet_fifo_source_source_valid = (rx_packet_fifo_out_valid & rx_packet_fifo_param_fifo_source_valid);
assign rx_packet_fifo_param_fifo_source_ready = ((rx_packet_fifo_source_source_valid & rx_packet_fifo_source_source_last) & rx_packet_fifo_source_source_ready);
assign rx_packet_fifo_param_fifo_syncfifo_din = {rx_packet_fifo_param_fifo_fifo_in_last, rx_packet_fifo_param_fifo_fifo_in_first, rx_packet_fifo_param_fifo_fifo_in_param_dummy};
assign {rx_packet_fifo_param_fifo_fifo_out_last, rx_packet_fifo_param_fifo_fifo_out_first, rx_packet_fifo_param_fifo_fifo_out_param_dummy} = rx_packet_fifo_param_fifo_syncfifo_dout;
assign rx_packet_fifo_param_fifo_sink_ready = rx_packet_fifo_param_fifo_syncfifo_writable;
assign rx_packet_fifo_param_fifo_syncfifo_we = rx_packet_fifo_param_fifo_sink_valid;
assign rx_packet_fifo_param_fifo_fifo_in_first = rx_packet_fifo_param_fifo_sink_first;
assign rx_packet_fifo_param_fifo_fifo_in_last = rx_packet_fifo_param_fifo_sink_last;
assign rx_packet_fifo_param_fifo_fifo_in_param_dummy = rx_packet_fifo_param_fifo_sink_param_dummy;
assign rx_packet_fifo_param_fifo_source_valid = rx_packet_fifo_param_fifo_syncfifo_readable;
assign rx_packet_fifo_param_fifo_source_first = rx_packet_fifo_param_fifo_fifo_out_first;
assign rx_packet_fifo_param_fifo_source_last = rx_packet_fifo_param_fifo_fifo_out_last;
assign rx_packet_fifo_param_fifo_source_param_dummy = rx_packet_fifo_param_fifo_fifo_out_param_dummy;
assign rx_packet_fifo_param_fifo_syncfifo_re = rx_packet_fifo_param_fifo_source_ready;

// synthesis translate_off
reg dummy_d_11;
// synthesis translate_on
always @(*) begin
	rx_packet_fifo_param_fifo_wrport_adr <= 2'd0;
	if (rx_packet_fifo_param_fifo_replace) begin
		rx_packet_fifo_param_fifo_wrport_adr <= (rx_packet_fifo_param_fifo_produce - 1'd1);
	end else begin
		rx_packet_fifo_param_fifo_wrport_adr <= rx_packet_fifo_param_fifo_produce;
	end
// synthesis translate_off
	dummy_d_11 <= dummy_s;
// synthesis translate_on
end
assign rx_packet_fifo_param_fifo_wrport_dat_w = rx_packet_fifo_param_fifo_syncfifo_din;
assign rx_packet_fifo_param_fifo_wrport_we = (rx_packet_fifo_param_fifo_syncfifo_we & (rx_packet_fifo_param_fifo_syncfifo_writable | rx_packet_fifo_param_fifo_replace));
assign rx_packet_fifo_param_fifo_do_read = (rx_packet_fifo_param_fifo_syncfifo_readable & rx_packet_fifo_param_fifo_syncfifo_re);
assign rx_packet_fifo_param_fifo_rdport_adr = rx_packet_fifo_param_fifo_consume;
assign rx_packet_fifo_param_fifo_syncfifo_dout = rx_packet_fifo_param_fifo_rdport_dat_r;
assign rx_packet_fifo_param_fifo_syncfifo_writable = (rx_packet_fifo_param_fifo_level != 3'd4);
assign rx_packet_fifo_param_fifo_syncfifo_readable = (rx_packet_fifo_param_fifo_level != 1'd0);
assign o = (toggle_o ^ toggle_o_r);
assign rx_cdc_write_rst = (eth_rx_rst | rx_cdc_write_ext_rst);
assign rx_cdc_cdc_reset_write = rx_cdc_write_ext_rst;
assign rx_cdc_cdc_reset_read = rx_cdc_ext_rst_to;
assign rx_cdc_cdc_sink_valid = rx_cdc_sink_sink_valid;
assign rx_cdc_sink_sink_ready = rx_cdc_cdc_sink_ready;
assign rx_cdc_cdc_sink_first = rx_cdc_sink_sink_first;
assign rx_cdc_cdc_sink_last = rx_cdc_sink_sink_last;
assign rx_cdc_cdc_sink_payload_data = rx_cdc_sink_sink_payload_data;
assign rx_cdc_cdc_sink_payload_be = rx_cdc_sink_sink_payload_be;
assign rx_cdc_cdc_sink_payload_error = rx_cdc_sink_sink_payload_error;
assign rx_cdc_cdc_source_ready = rx_cdc_source_source_ready;
assign rx_cdc_source_source_first = rx_cdc_cdc_source_first;
assign rx_cdc_source_source_last = rx_cdc_cdc_source_last;
assign rx_cdc_source_source_payload_data = rx_cdc_cdc_source_payload_data;
assign rx_cdc_source_source_payload_be = rx_cdc_cdc_source_payload_be;
assign rx_cdc_source_source_payload_error = rx_cdc_cdc_source_payload_error;
assign rx_cdc_source_source_valid = ((rx_cdc_cdc_source_valid & (~rx_cdc_ext_rst_to)) & (~sys_rst));
assign rx_cdc_cdc_asyncfifo_din = {rx_cdc_cdc_fifo_in_last, rx_cdc_cdc_fifo_in_first, rx_cdc_cdc_fifo_in_payload_error, rx_cdc_cdc_fifo_in_payload_be, rx_cdc_cdc_fifo_in_payload_data};
assign {rx_cdc_cdc_fifo_out_last, rx_cdc_cdc_fifo_out_first, rx_cdc_cdc_fifo_out_payload_error, rx_cdc_cdc_fifo_out_payload_be, rx_cdc_cdc_fifo_out_payload_data} = rx_cdc_cdc_asyncfifo_dout;
assign rx_cdc_cdc_sink_ready = rx_cdc_cdc_asyncfifo_writable;
assign rx_cdc_cdc_asyncfifo_we = rx_cdc_cdc_sink_valid;
assign rx_cdc_cdc_fifo_in_first = rx_cdc_cdc_sink_first;
assign rx_cdc_cdc_fifo_in_last = rx_cdc_cdc_sink_last;
assign rx_cdc_cdc_fifo_in_payload_data = rx_cdc_cdc_sink_payload_data;
assign rx_cdc_cdc_fifo_in_payload_be = rx_cdc_cdc_sink_payload_be;
assign rx_cdc_cdc_fifo_in_payload_error = rx_cdc_cdc_sink_payload_error;
assign rx_cdc_cdc_source_valid = rx_cdc_cdc_asyncfifo_readable;
assign rx_cdc_cdc_source_first = rx_cdc_cdc_fifo_out_first;
assign rx_cdc_cdc_source_last = rx_cdc_cdc_fifo_out_last;
assign rx_cdc_cdc_source_payload_data = rx_cdc_cdc_fifo_out_payload_data;
assign rx_cdc_cdc_source_payload_be = rx_cdc_cdc_fifo_out_payload_be;
assign rx_cdc_cdc_source_payload_error = rx_cdc_cdc_fifo_out_payload_error;
assign rx_cdc_cdc_asyncfifo_re = rx_cdc_cdc_source_ready;
assign rx_cdc_cdc_graycounter0_ce = (rx_cdc_cdc_asyncfifo_writable & rx_cdc_cdc_asyncfifo_we);
assign rx_cdc_cdc_graycounter1_ce = (rx_cdc_cdc_asyncfifo_readable & rx_cdc_cdc_asyncfifo_re);
assign rx_cdc_cdc_asyncfifo_writable = (((rx_cdc_cdc_graycounter0_q[5] == rx_cdc_cdc_consume_wdomain[5]) | (rx_cdc_cdc_graycounter0_q[4] == rx_cdc_cdc_consume_wdomain[4])) | (rx_cdc_cdc_graycounter0_q[3:0] != rx_cdc_cdc_consume_wdomain[3:0]));
assign rx_cdc_cdc_asyncfifo_readable = (rx_cdc_cdc_graycounter1_q != rx_cdc_cdc_produce_rdomain);
assign rx_cdc_cdc_wrport_adr = rx_cdc_cdc_graycounter0_q_binary[4:0];
assign rx_cdc_cdc_wrport_dat_w = rx_cdc_cdc_asyncfifo_din;
assign rx_cdc_cdc_wrport_we = rx_cdc_cdc_graycounter0_ce;
assign rx_cdc_cdc_rdport_adr = rx_cdc_cdc_graycounter1_q_next_binary[4:0];
assign rx_cdc_cdc_asyncfifo_dout = rx_cdc_cdc_rdport_dat_r;

// synthesis translate_off
reg dummy_d_12;
// synthesis translate_on
always @(*) begin
	rx_cdc_cdc_graycounter0_q_next_binary <= 6'd0;
	if (rx_cdc_cdc_graycounter0_ce) begin
		rx_cdc_cdc_graycounter0_q_next_binary <= (rx_cdc_cdc_graycounter0_q_binary + 1'd1);
	end else begin
		rx_cdc_cdc_graycounter0_q_next_binary <= rx_cdc_cdc_graycounter0_q_binary;
	end
// synthesis translate_off
	dummy_d_12 <= dummy_s;
// synthesis translate_on
end
assign rx_cdc_cdc_graycounter0_q_next = (rx_cdc_cdc_graycounter0_q_next_binary ^ rx_cdc_cdc_graycounter0_q_next_binary[5:1]);

// synthesis translate_off
reg dummy_d_13;
// synthesis translate_on
always @(*) begin
	rx_cdc_cdc_graycounter1_q_next_binary <= 6'd0;
	if (rx_cdc_cdc_graycounter1_ce) begin
		rx_cdc_cdc_graycounter1_q_next_binary <= (rx_cdc_cdc_graycounter1_q_binary + 1'd1);
	end else begin
		rx_cdc_cdc_graycounter1_q_next_binary <= rx_cdc_cdc_graycounter1_q_binary;
	end
// synthesis translate_off
	dummy_d_13 <= dummy_s;
// synthesis translate_on
end
assign rx_cdc_cdc_graycounter1_q_next = (rx_cdc_cdc_graycounter1_q_next_binary ^ rx_cdc_cdc_graycounter1_q_next_binary[5:1]);
assign rx_preamble_source_payload_data = rx_preamble_sink_payload_data;
assign rx_preamble_source_payload_be = rx_preamble_sink_payload_be;

// synthesis translate_off
reg dummy_d_14;
// synthesis translate_on
always @(*) begin
	rx_preamble_sink_ready <= 1'd0;
	rx_preamble_source_valid <= 1'd0;
	rx_preamble_source_first <= 1'd0;
	rx_preamble_source_last <= 1'd0;
	rx_preamble_source_payload_error <= 1'd0;
	rx_preamble_error <= 1'd0;
	rxdatapath_liteethmacpreamblechecker_next_state <= 1'd0;
	rxdatapath_liteethmacpreamblechecker_next_state <= rxdatapath_liteethmacpreamblechecker_state;
	case (rxdatapath_liteethmacpreamblechecker_state)
		1'd1: begin
			rx_preamble_source_valid <= rx_preamble_sink_valid;
			rx_preamble_sink_ready <= rx_preamble_source_ready;
			rx_preamble_source_first <= rx_preamble_sink_first;
			rx_preamble_source_last <= rx_preamble_sink_last;
			rx_preamble_source_payload_error <= rx_preamble_sink_payload_error;
			if (((rx_preamble_source_valid & rx_preamble_source_last) & rx_preamble_source_ready)) begin
				rxdatapath_liteethmacpreamblechecker_next_state <= 1'd0;
			end
		end
		default: begin
			rx_preamble_sink_ready <= 1'd1;
			if (((rx_preamble_sink_valid & (~rx_preamble_sink_last)) & (rx_preamble_sink_payload_data == rx_preamble_preamble[63:56]))) begin
				rxdatapath_liteethmacpreamblechecker_next_state <= 1'd1;
			end
			if ((rx_preamble_sink_valid & rx_preamble_sink_last)) begin
				rx_preamble_error <= 1'd1;
			end
		end
	endcase
// synthesis translate_off
	dummy_d_14 <= dummy_s;
// synthesis translate_on
end
assign pulsesynchronizer0_o = (pulsesynchronizer0_toggle_o ^ pulsesynchronizer0_toggle_o_r);
assign liteethmaccrc32checker_fifo_full = (liteethmaccrc32checker_syncfifo_level == 3'd4);
assign liteethmaccrc32checker_fifo_out = (liteethmaccrc32checker_source_source_valid & liteethmaccrc32checker_source_source_ready);
assign liteethmaccrc32checker_syncfifo_sink_first = liteethmaccrc32checker_sink_sink_first;
assign liteethmaccrc32checker_syncfifo_sink_last = liteethmaccrc32checker_sink_sink_last;
assign liteethmaccrc32checker_syncfifo_sink_payload_data = liteethmaccrc32checker_sink_sink_payload_data;
assign liteethmaccrc32checker_syncfifo_sink_payload_be = liteethmaccrc32checker_sink_sink_payload_be;
assign liteethmaccrc32checker_syncfifo_sink_payload_error = liteethmaccrc32checker_sink_sink_payload_error;

// synthesis translate_off
reg dummy_d_15;
// synthesis translate_on
always @(*) begin
	liteethmaccrc32checker_syncfifo_sink_valid <= 1'd0;
	liteethmaccrc32checker_syncfifo_sink_valid <= liteethmaccrc32checker_sink_sink_valid;
	liteethmaccrc32checker_syncfifo_sink_valid <= liteethmaccrc32checker_fifo_in;
// synthesis translate_off
	dummy_d_15 <= dummy_s;
// synthesis translate_on
end

// synthesis translate_off
reg dummy_d_16;
// synthesis translate_on
always @(*) begin
	liteethmaccrc32checker_sink_sink_ready <= 1'd0;
	liteethmaccrc32checker_sink_sink_ready <= liteethmaccrc32checker_syncfifo_sink_ready;
	liteethmaccrc32checker_sink_sink_ready <= liteethmaccrc32checker_fifo_in;
// synthesis translate_off
	dummy_d_16 <= dummy_s;
// synthesis translate_on
end
assign liteethmaccrc32checker_fifo_in = (((liteethmaccrc32checker_sink_sink_valid & ((~liteethmaccrc32checker_fifo_full) | liteethmaccrc32checker_fifo_out)) & (~liteethmaccrc32checker_is_ongoing0)) & (~liteethmaccrc32checker_is_ongoing1));
assign liteethmaccrc32checker_crc_data0 = liteethmaccrc32checker_sink_sink_payload_data;
assign liteethmaccrc32checker_crc_be = liteethmaccrc32checker_sink_sink_payload_be;
assign liteethmaccrc32checker_last_word_error = ((liteethmaccrc32checker_crc_error | liteethmaccrc32checker_packet_error) | ((liteethmaccrc32checker_sink_sink_payload_error & liteethmaccrc32checker_sink_sink_payload_be) != 1'd0));
assign liteethmaccrc32checker_source_source_first = liteethmaccrc32checker_syncfifo_source_first;
assign liteethmaccrc32checker_source_source_payload_data = liteethmaccrc32checker_syncfifo_source_payload_data;
assign liteethmaccrc32checker_sink_sink_valid = bufferizeendpoints_source_source_valid;
assign bufferizeendpoints_source_source_ready = liteethmaccrc32checker_sink_sink_ready;
assign liteethmaccrc32checker_sink_sink_first = bufferizeendpoints_source_source_first;
assign liteethmaccrc32checker_sink_sink_last = bufferizeendpoints_source_source_last;
assign liteethmaccrc32checker_sink_sink_payload_data = bufferizeendpoints_source_source_payload_data;
assign liteethmaccrc32checker_sink_sink_payload_be = bufferizeendpoints_source_source_payload_be;
assign liteethmaccrc32checker_sink_sink_payload_error = bufferizeendpoints_source_source_payload_error;
assign liteethmaccrc32checker_crc_crc_prev = liteethmaccrc32checker_crc_reg;

// synthesis translate_off
reg dummy_d_17;
// synthesis translate_on
always @(*) begin
	liteethmaccrc32checker_crc_error <= 1'd0;
	liteethmaccrc32checker_crc_data1 <= 8'd0;
	liteethmaccrc32checker_crc_data1 <= liteethmaccrc32checker_crc_data0;
	if (liteethmaccrc32checker_crc_be) begin
		liteethmaccrc32checker_crc_data1 <= (liteethmaccrc32checker_crc_data0 & 8'd255);
		liteethmaccrc32checker_crc_error <= (liteethmaccrc32checker_crc_crc_next != 32'd3338984827);
	end
// synthesis translate_off
	dummy_d_17 <= dummy_s;
// synthesis translate_on
end

// synthesis translate_off
reg dummy_d_18;
// synthesis translate_on
always @(*) begin
	liteethmaccrc32checker_crc_crc_next <= 32'd0;
	liteethmaccrc32checker_crc_crc_next[0] <= (((liteethmaccrc32checker_crc_crc_prev[24] ^ liteethmaccrc32checker_crc_crc_prev[30]) ^ liteethmaccrc32checker_crc_data1[1]) ^ liteethmaccrc32checker_crc_data1[7]);
	liteethmaccrc32checker_crc_crc_next[1] <= (((((((liteethmaccrc32checker_crc_crc_prev[25] ^ liteethmaccrc32checker_crc_crc_prev[31]) ^ liteethmaccrc32checker_crc_data1[0]) ^ liteethmaccrc32checker_crc_data1[6]) ^ liteethmaccrc32checker_crc_crc_prev[24]) ^ liteethmaccrc32checker_crc_crc_prev[30]) ^ liteethmaccrc32checker_crc_data1[1]) ^ liteethmaccrc32checker_crc_data1[7]);
	liteethmaccrc32checker_crc_crc_next[2] <= (((((((((liteethmaccrc32checker_crc_crc_prev[26] ^ liteethmaccrc32checker_crc_data1[5]) ^ liteethmaccrc32checker_crc_crc_prev[25]) ^ liteethmaccrc32checker_crc_crc_prev[31]) ^ liteethmaccrc32checker_crc_data1[0]) ^ liteethmaccrc32checker_crc_data1[6]) ^ liteethmaccrc32checker_crc_crc_prev[24]) ^ liteethmaccrc32checker_crc_crc_prev[30]) ^ liteethmaccrc32checker_crc_data1[1]) ^ liteethmaccrc32checker_crc_data1[7]);
	liteethmaccrc32checker_crc_crc_next[3] <= (((((((liteethmaccrc32checker_crc_crc_prev[27] ^ liteethmaccrc32checker_crc_data1[4]) ^ liteethmaccrc32checker_crc_crc_prev[26]) ^ liteethmaccrc32checker_crc_data1[5]) ^ liteethmaccrc32checker_crc_crc_prev[25]) ^ liteethmaccrc32checker_crc_crc_prev[31]) ^ liteethmaccrc32checker_crc_data1[0]) ^ liteethmaccrc32checker_crc_data1[6]);
	liteethmaccrc32checker_crc_crc_next[4] <= (((((((((liteethmaccrc32checker_crc_crc_prev[28] ^ liteethmaccrc32checker_crc_data1[3]) ^ liteethmaccrc32checker_crc_crc_prev[27]) ^ liteethmaccrc32checker_crc_data1[4]) ^ liteethmaccrc32checker_crc_crc_prev[26]) ^ liteethmaccrc32checker_crc_data1[5]) ^ liteethmaccrc32checker_crc_crc_prev[24]) ^ liteethmaccrc32checker_crc_crc_prev[30]) ^ liteethmaccrc32checker_crc_data1[1]) ^ liteethmaccrc32checker_crc_data1[7]);
	liteethmaccrc32checker_crc_crc_next[5] <= (((((((((((((liteethmaccrc32checker_crc_crc_prev[29] ^ liteethmaccrc32checker_crc_data1[2]) ^ liteethmaccrc32checker_crc_crc_prev[28]) ^ liteethmaccrc32checker_crc_data1[3]) ^ liteethmaccrc32checker_crc_crc_prev[27]) ^ liteethmaccrc32checker_crc_data1[4]) ^ liteethmaccrc32checker_crc_crc_prev[25]) ^ liteethmaccrc32checker_crc_crc_prev[31]) ^ liteethmaccrc32checker_crc_data1[0]) ^ liteethmaccrc32checker_crc_data1[6]) ^ liteethmaccrc32checker_crc_crc_prev[24]) ^ liteethmaccrc32checker_crc_crc_prev[30]) ^ liteethmaccrc32checker_crc_data1[1]) ^ liteethmaccrc32checker_crc_data1[7]);
	liteethmaccrc32checker_crc_crc_next[6] <= (((((((((((liteethmaccrc32checker_crc_crc_prev[30] ^ liteethmaccrc32checker_crc_data1[1]) ^ liteethmaccrc32checker_crc_crc_prev[29]) ^ liteethmaccrc32checker_crc_data1[2]) ^ liteethmaccrc32checker_crc_crc_prev[28]) ^ liteethmaccrc32checker_crc_data1[3]) ^ liteethmaccrc32checker_crc_crc_prev[26]) ^ liteethmaccrc32checker_crc_data1[5]) ^ liteethmaccrc32checker_crc_crc_prev[25]) ^ liteethmaccrc32checker_crc_crc_prev[31]) ^ liteethmaccrc32checker_crc_data1[0]) ^ liteethmaccrc32checker_crc_data1[6]);
	liteethmaccrc32checker_crc_crc_next[7] <= (((((((((liteethmaccrc32checker_crc_crc_prev[31] ^ liteethmaccrc32checker_crc_data1[0]) ^ liteethmaccrc32checker_crc_crc_prev[29]) ^ liteethmaccrc32checker_crc_data1[2]) ^ liteethmaccrc32checker_crc_crc_prev[27]) ^ liteethmaccrc32checker_crc_data1[4]) ^ liteethmaccrc32checker_crc_crc_prev[26]) ^ liteethmaccrc32checker_crc_data1[5]) ^ liteethmaccrc32checker_crc_crc_prev[24]) ^ liteethmaccrc32checker_crc_data1[7]);
	liteethmaccrc32checker_crc_crc_next[8] <= ((((((((liteethmaccrc32checker_crc_crc_prev[0] ^ liteethmaccrc32checker_crc_crc_prev[28]) ^ liteethmaccrc32checker_crc_data1[3]) ^ liteethmaccrc32checker_crc_crc_prev[27]) ^ liteethmaccrc32checker_crc_data1[4]) ^ liteethmaccrc32checker_crc_crc_prev[25]) ^ liteethmaccrc32checker_crc_data1[6]) ^ liteethmaccrc32checker_crc_crc_prev[24]) ^ liteethmaccrc32checker_crc_data1[7]);
	liteethmaccrc32checker_crc_crc_next[9] <= ((((((((liteethmaccrc32checker_crc_crc_prev[1] ^ liteethmaccrc32checker_crc_crc_prev[29]) ^ liteethmaccrc32checker_crc_data1[2]) ^ liteethmaccrc32checker_crc_crc_prev[28]) ^ liteethmaccrc32checker_crc_data1[3]) ^ liteethmaccrc32checker_crc_crc_prev[26]) ^ liteethmaccrc32checker_crc_data1[5]) ^ liteethmaccrc32checker_crc_crc_prev[25]) ^ liteethmaccrc32checker_crc_data1[6]);
	liteethmaccrc32checker_crc_crc_next[10] <= ((((((((liteethmaccrc32checker_crc_crc_prev[2] ^ liteethmaccrc32checker_crc_crc_prev[29]) ^ liteethmaccrc32checker_crc_data1[2]) ^ liteethmaccrc32checker_crc_crc_prev[27]) ^ liteethmaccrc32checker_crc_data1[4]) ^ liteethmaccrc32checker_crc_crc_prev[26]) ^ liteethmaccrc32checker_crc_data1[5]) ^ liteethmaccrc32checker_crc_crc_prev[24]) ^ liteethmaccrc32checker_crc_data1[7]);
	liteethmaccrc32checker_crc_crc_next[11] <= ((((((((liteethmaccrc32checker_crc_crc_prev[3] ^ liteethmaccrc32checker_crc_crc_prev[28]) ^ liteethmaccrc32checker_crc_data1[3]) ^ liteethmaccrc32checker_crc_crc_prev[27]) ^ liteethmaccrc32checker_crc_data1[4]) ^ liteethmaccrc32checker_crc_crc_prev[25]) ^ liteethmaccrc32checker_crc_data1[6]) ^ liteethmaccrc32checker_crc_crc_prev[24]) ^ liteethmaccrc32checker_crc_data1[7]);
	liteethmaccrc32checker_crc_crc_next[12] <= ((((((((((((liteethmaccrc32checker_crc_crc_prev[4] ^ liteethmaccrc32checker_crc_crc_prev[29]) ^ liteethmaccrc32checker_crc_data1[2]) ^ liteethmaccrc32checker_crc_crc_prev[28]) ^ liteethmaccrc32checker_crc_data1[3]) ^ liteethmaccrc32checker_crc_crc_prev[26]) ^ liteethmaccrc32checker_crc_data1[5]) ^ liteethmaccrc32checker_crc_crc_prev[25]) ^ liteethmaccrc32checker_crc_data1[6]) ^ liteethmaccrc32checker_crc_crc_prev[24]) ^ liteethmaccrc32checker_crc_crc_prev[30]) ^ liteethmaccrc32checker_crc_data1[1]) ^ liteethmaccrc32checker_crc_data1[7]);
	liteethmaccrc32checker_crc_crc_next[13] <= ((((((((((((liteethmaccrc32checker_crc_crc_prev[5] ^ liteethmaccrc32checker_crc_crc_prev[30]) ^ liteethmaccrc32checker_crc_data1[1]) ^ liteethmaccrc32checker_crc_crc_prev[29]) ^ liteethmaccrc32checker_crc_data1[2]) ^ liteethmaccrc32checker_crc_crc_prev[27]) ^ liteethmaccrc32checker_crc_data1[4]) ^ liteethmaccrc32checker_crc_crc_prev[26]) ^ liteethmaccrc32checker_crc_data1[5]) ^ liteethmaccrc32checker_crc_crc_prev[25]) ^ liteethmaccrc32checker_crc_crc_prev[31]) ^ liteethmaccrc32checker_crc_data1[0]) ^ liteethmaccrc32checker_crc_data1[6]);
	liteethmaccrc32checker_crc_crc_next[14] <= ((((((((((liteethmaccrc32checker_crc_crc_prev[6] ^ liteethmaccrc32checker_crc_crc_prev[31]) ^ liteethmaccrc32checker_crc_data1[0]) ^ liteethmaccrc32checker_crc_crc_prev[30]) ^ liteethmaccrc32checker_crc_data1[1]) ^ liteethmaccrc32checker_crc_crc_prev[28]) ^ liteethmaccrc32checker_crc_data1[3]) ^ liteethmaccrc32checker_crc_crc_prev[27]) ^ liteethmaccrc32checker_crc_data1[4]) ^ liteethmaccrc32checker_crc_crc_prev[26]) ^ liteethmaccrc32checker_crc_data1[5]);
	liteethmaccrc32checker_crc_crc_next[15] <= ((((((((liteethmaccrc32checker_crc_crc_prev[7] ^ liteethmaccrc32checker_crc_crc_prev[31]) ^ liteethmaccrc32checker_crc_data1[0]) ^ liteethmaccrc32checker_crc_crc_prev[29]) ^ liteethmaccrc32checker_crc_data1[2]) ^ liteethmaccrc32checker_crc_crc_prev[28]) ^ liteethmaccrc32checker_crc_data1[3]) ^ liteethmaccrc32checker_crc_crc_prev[27]) ^ liteethmaccrc32checker_crc_data1[4]);
	liteethmaccrc32checker_crc_crc_next[16] <= ((((((liteethmaccrc32checker_crc_crc_prev[8] ^ liteethmaccrc32checker_crc_crc_prev[29]) ^ liteethmaccrc32checker_crc_data1[2]) ^ liteethmaccrc32checker_crc_crc_prev[28]) ^ liteethmaccrc32checker_crc_data1[3]) ^ liteethmaccrc32checker_crc_crc_prev[24]) ^ liteethmaccrc32checker_crc_data1[7]);
	liteethmaccrc32checker_crc_crc_next[17] <= ((((((liteethmaccrc32checker_crc_crc_prev[9] ^ liteethmaccrc32checker_crc_crc_prev[30]) ^ liteethmaccrc32checker_crc_data1[1]) ^ liteethmaccrc32checker_crc_crc_prev[29]) ^ liteethmaccrc32checker_crc_data1[2]) ^ liteethmaccrc32checker_crc_crc_prev[25]) ^ liteethmaccrc32checker_crc_data1[6]);
	liteethmaccrc32checker_crc_crc_next[18] <= ((((((liteethmaccrc32checker_crc_crc_prev[10] ^ liteethmaccrc32checker_crc_crc_prev[31]) ^ liteethmaccrc32checker_crc_data1[0]) ^ liteethmaccrc32checker_crc_crc_prev[30]) ^ liteethmaccrc32checker_crc_data1[1]) ^ liteethmaccrc32checker_crc_crc_prev[26]) ^ liteethmaccrc32checker_crc_data1[5]);
	liteethmaccrc32checker_crc_crc_next[19] <= ((((liteethmaccrc32checker_crc_crc_prev[11] ^ liteethmaccrc32checker_crc_crc_prev[31]) ^ liteethmaccrc32checker_crc_data1[0]) ^ liteethmaccrc32checker_crc_crc_prev[27]) ^ liteethmaccrc32checker_crc_data1[4]);
	liteethmaccrc32checker_crc_crc_next[20] <= ((liteethmaccrc32checker_crc_crc_prev[12] ^ liteethmaccrc32checker_crc_crc_prev[28]) ^ liteethmaccrc32checker_crc_data1[3]);
	liteethmaccrc32checker_crc_crc_next[21] <= ((liteethmaccrc32checker_crc_crc_prev[13] ^ liteethmaccrc32checker_crc_crc_prev[29]) ^ liteethmaccrc32checker_crc_data1[2]);
	liteethmaccrc32checker_crc_crc_next[22] <= ((liteethmaccrc32checker_crc_crc_prev[14] ^ liteethmaccrc32checker_crc_crc_prev[24]) ^ liteethmaccrc32checker_crc_data1[7]);
	liteethmaccrc32checker_crc_crc_next[23] <= ((((((liteethmaccrc32checker_crc_crc_prev[15] ^ liteethmaccrc32checker_crc_crc_prev[25]) ^ liteethmaccrc32checker_crc_data1[6]) ^ liteethmaccrc32checker_crc_crc_prev[24]) ^ liteethmaccrc32checker_crc_crc_prev[30]) ^ liteethmaccrc32checker_crc_data1[1]) ^ liteethmaccrc32checker_crc_data1[7]);
	liteethmaccrc32checker_crc_crc_next[24] <= ((((((liteethmaccrc32checker_crc_crc_prev[16] ^ liteethmaccrc32checker_crc_crc_prev[26]) ^ liteethmaccrc32checker_crc_data1[5]) ^ liteethmaccrc32checker_crc_crc_prev[25]) ^ liteethmaccrc32checker_crc_crc_prev[31]) ^ liteethmaccrc32checker_crc_data1[0]) ^ liteethmaccrc32checker_crc_data1[6]);
	liteethmaccrc32checker_crc_crc_next[25] <= ((((liteethmaccrc32checker_crc_crc_prev[17] ^ liteethmaccrc32checker_crc_crc_prev[27]) ^ liteethmaccrc32checker_crc_data1[4]) ^ liteethmaccrc32checker_crc_crc_prev[26]) ^ liteethmaccrc32checker_crc_data1[5]);
	liteethmaccrc32checker_crc_crc_next[26] <= ((((((((liteethmaccrc32checker_crc_crc_prev[18] ^ liteethmaccrc32checker_crc_crc_prev[28]) ^ liteethmaccrc32checker_crc_data1[3]) ^ liteethmaccrc32checker_crc_crc_prev[27]) ^ liteethmaccrc32checker_crc_data1[4]) ^ liteethmaccrc32checker_crc_crc_prev[24]) ^ liteethmaccrc32checker_crc_crc_prev[30]) ^ liteethmaccrc32checker_crc_data1[1]) ^ liteethmaccrc32checker_crc_data1[7]);
	liteethmaccrc32checker_crc_crc_next[27] <= ((((((((liteethmaccrc32checker_crc_crc_prev[19] ^ liteethmaccrc32checker_crc_crc_prev[29]) ^ liteethmaccrc32checker_crc_data1[2]) ^ liteethmaccrc32checker_crc_crc_prev[28]) ^ liteethmaccrc32checker_crc_data1[3]) ^ liteethmaccrc32checker_crc_crc_prev[25]) ^ liteethmaccrc32checker_crc_crc_prev[31]) ^ liteethmaccrc32checker_crc_data1[0]) ^ liteethmaccrc32checker_crc_data1[6]);
	liteethmaccrc32checker_crc_crc_next[28] <= ((((((liteethmaccrc32checker_crc_crc_prev[20] ^ liteethmaccrc32checker_crc_crc_prev[30]) ^ liteethmaccrc32checker_crc_data1[1]) ^ liteethmaccrc32checker_crc_crc_prev[29]) ^ liteethmaccrc32checker_crc_data1[2]) ^ liteethmaccrc32checker_crc_crc_prev[26]) ^ liteethmaccrc32checker_crc_data1[5]);
	liteethmaccrc32checker_crc_crc_next[29] <= ((((((liteethmaccrc32checker_crc_crc_prev[21] ^ liteethmaccrc32checker_crc_crc_prev[31]) ^ liteethmaccrc32checker_crc_data1[0]) ^ liteethmaccrc32checker_crc_crc_prev[30]) ^ liteethmaccrc32checker_crc_data1[1]) ^ liteethmaccrc32checker_crc_crc_prev[27]) ^ liteethmaccrc32checker_crc_data1[4]);
	liteethmaccrc32checker_crc_crc_next[30] <= ((((liteethmaccrc32checker_crc_crc_prev[22] ^ liteethmaccrc32checker_crc_crc_prev[31]) ^ liteethmaccrc32checker_crc_data1[0]) ^ liteethmaccrc32checker_crc_crc_prev[28]) ^ liteethmaccrc32checker_crc_data1[3]);
	liteethmaccrc32checker_crc_crc_next[31] <= ((liteethmaccrc32checker_crc_crc_prev[23] ^ liteethmaccrc32checker_crc_crc_prev[29]) ^ liteethmaccrc32checker_crc_data1[2]);
// synthesis translate_off
	dummy_d_18 <= dummy_s;
// synthesis translate_on
end
assign liteethmaccrc32checker_syncfifo_syncfifo_din = {liteethmaccrc32checker_syncfifo_fifo_in_last, liteethmaccrc32checker_syncfifo_fifo_in_first, liteethmaccrc32checker_syncfifo_fifo_in_payload_error, liteethmaccrc32checker_syncfifo_fifo_in_payload_be, liteethmaccrc32checker_syncfifo_fifo_in_payload_data};
assign {liteethmaccrc32checker_syncfifo_fifo_out_last, liteethmaccrc32checker_syncfifo_fifo_out_first, liteethmaccrc32checker_syncfifo_fifo_out_payload_error, liteethmaccrc32checker_syncfifo_fifo_out_payload_be, liteethmaccrc32checker_syncfifo_fifo_out_payload_data} = liteethmaccrc32checker_syncfifo_syncfifo_dout;
assign liteethmaccrc32checker_syncfifo_sink_ready = liteethmaccrc32checker_syncfifo_syncfifo_writable;
assign liteethmaccrc32checker_syncfifo_syncfifo_we = liteethmaccrc32checker_syncfifo_sink_valid;
assign liteethmaccrc32checker_syncfifo_fifo_in_first = liteethmaccrc32checker_syncfifo_sink_first;
assign liteethmaccrc32checker_syncfifo_fifo_in_last = liteethmaccrc32checker_syncfifo_sink_last;
assign liteethmaccrc32checker_syncfifo_fifo_in_payload_data = liteethmaccrc32checker_syncfifo_sink_payload_data;
assign liteethmaccrc32checker_syncfifo_fifo_in_payload_be = liteethmaccrc32checker_syncfifo_sink_payload_be;
assign liteethmaccrc32checker_syncfifo_fifo_in_payload_error = liteethmaccrc32checker_syncfifo_sink_payload_error;
assign liteethmaccrc32checker_syncfifo_source_valid = liteethmaccrc32checker_syncfifo_syncfifo_readable;
assign liteethmaccrc32checker_syncfifo_source_first = liteethmaccrc32checker_syncfifo_fifo_out_first;
assign liteethmaccrc32checker_syncfifo_source_last = liteethmaccrc32checker_syncfifo_fifo_out_last;
assign liteethmaccrc32checker_syncfifo_source_payload_data = liteethmaccrc32checker_syncfifo_fifo_out_payload_data;
assign liteethmaccrc32checker_syncfifo_source_payload_be = liteethmaccrc32checker_syncfifo_fifo_out_payload_be;
assign liteethmaccrc32checker_syncfifo_source_payload_error = liteethmaccrc32checker_syncfifo_fifo_out_payload_error;
assign liteethmaccrc32checker_syncfifo_syncfifo_re = liteethmaccrc32checker_syncfifo_source_ready;

// synthesis translate_off
reg dummy_d_19;
// synthesis translate_on
always @(*) begin
	liteethmaccrc32checker_syncfifo_wrport_adr <= 3'd0;
	if (liteethmaccrc32checker_syncfifo_replace) begin
		liteethmaccrc32checker_syncfifo_wrport_adr <= (liteethmaccrc32checker_syncfifo_produce - 1'd1);
	end else begin
		liteethmaccrc32checker_syncfifo_wrport_adr <= liteethmaccrc32checker_syncfifo_produce;
	end
// synthesis translate_off
	dummy_d_19 <= dummy_s;
// synthesis translate_on
end
assign liteethmaccrc32checker_syncfifo_wrport_dat_w = liteethmaccrc32checker_syncfifo_syncfifo_din;
assign liteethmaccrc32checker_syncfifo_wrport_we = (liteethmaccrc32checker_syncfifo_syncfifo_we & (liteethmaccrc32checker_syncfifo_syncfifo_writable | liteethmaccrc32checker_syncfifo_replace));
assign liteethmaccrc32checker_syncfifo_do_read = (liteethmaccrc32checker_syncfifo_syncfifo_readable & liteethmaccrc32checker_syncfifo_syncfifo_re);
assign liteethmaccrc32checker_syncfifo_rdport_adr = liteethmaccrc32checker_syncfifo_consume;
assign liteethmaccrc32checker_syncfifo_syncfifo_dout = liteethmaccrc32checker_syncfifo_rdport_dat_r;
assign liteethmaccrc32checker_syncfifo_syncfifo_writable = (liteethmaccrc32checker_syncfifo_level != 3'd5);
assign liteethmaccrc32checker_syncfifo_syncfifo_readable = (liteethmaccrc32checker_syncfifo_level != 1'd0);

// synthesis translate_off
reg dummy_d_20;
// synthesis translate_on
always @(*) begin
	liteethmaccrc32checker_source_source_valid <= 1'd0;
	liteethmaccrc32checker_source_source_last <= 1'd0;
	liteethmaccrc32checker_source_source_payload_be <= 1'd0;
	liteethmaccrc32checker_source_source_payload_error <= 1'd0;
	liteethmaccrc32checker_error <= 1'd0;
	liteethmaccrc32checker_crc_ce <= 1'd0;
	liteethmaccrc32checker_crc_reset <= 1'd0;
	liteethmaccrc32checker_syncfifo_source_ready <= 1'd0;
	liteethmaccrc32checker_fifo_reset <= 1'd0;
	liteethmaccrc32checker_is_ongoing0 <= 1'd0;
	liteethmaccrc32checker_is_ongoing1 <= 1'd0;
	liteethmaccrc32checker_is_ongoing2 <= 1'd0;
	rxdatapath_bufferizeendpoints_next_state <= 2'd0;
	liteethmaccrc32checker_be_next_value0 <= 1'd0;
	liteethmaccrc32checker_be_next_value_ce0 <= 1'd0;
	liteethmaccrc32checker_last_error_next_value1 <= 1'd0;
	liteethmaccrc32checker_last_error_next_value_ce1 <= 1'd0;
	liteethmaccrc32checker_source_source_payload_error <= liteethmaccrc32checker_syncfifo_source_payload_error;
	rxdatapath_bufferizeendpoints_next_state <= rxdatapath_bufferizeendpoints_state;
	case (rxdatapath_bufferizeendpoints_state)
		1'd1: begin
			liteethmaccrc32checker_is_ongoing1 <= 1'd1;
			liteethmaccrc32checker_source_source_valid <= liteethmaccrc32checker_syncfifo_source_valid;
			liteethmaccrc32checker_syncfifo_source_ready <= liteethmaccrc32checker_source_source_ready;
			liteethmaccrc32checker_source_source_last <= liteethmaccrc32checker_syncfifo_source_last;
			liteethmaccrc32checker_source_source_payload_error <= (liteethmaccrc32checker_syncfifo_source_payload_error | {1{liteethmaccrc32checker_last_error}});
			liteethmaccrc32checker_source_source_payload_be <= liteethmaccrc32checker_be;
			if ((liteethmaccrc32checker_source_source_valid & liteethmaccrc32checker_source_source_ready)) begin
				rxdatapath_bufferizeendpoints_next_state <= 1'd0;
			end
		end
		2'd2: begin
			if ((liteethmaccrc32checker_sink_sink_valid & liteethmaccrc32checker_sink_sink_ready)) begin
				liteethmaccrc32checker_crc_ce <= 1'd1;
				rxdatapath_bufferizeendpoints_next_state <= 2'd3;
			end
		end
		2'd3: begin
			liteethmaccrc32checker_syncfifo_source_ready <= liteethmaccrc32checker_fifo_out;
			liteethmaccrc32checker_source_source_valid <= (liteethmaccrc32checker_sink_sink_valid & liteethmaccrc32checker_fifo_full);
			liteethmaccrc32checker_source_source_payload_be <= 1'd1;
			if (1'd1) begin
				liteethmaccrc32checker_source_source_last <= liteethmaccrc32checker_sink_sink_last;
				if (liteethmaccrc32checker_sink_sink_last) begin
					liteethmaccrc32checker_source_source_payload_be <= liteethmaccrc32checker_sink_sink_payload_be;
				end
			end else begin
				if ((liteethmaccrc32checker_sink_sink_payload_be <= 4'd15)) begin
					liteethmaccrc32checker_source_source_last <= liteethmaccrc32checker_sink_sink_last;
					liteethmaccrc32checker_source_source_payload_be <= ((liteethmaccrc32checker_sink_sink_payload_be <<< 3'd4) | 4'd15);
				end else begin
					liteethmaccrc32checker_be_next_value0 <= (liteethmaccrc32checker_sink_sink_payload_be >>> 3'd4);
					liteethmaccrc32checker_be_next_value_ce0 <= 1'd1;
					liteethmaccrc32checker_last_error_next_value1 <= liteethmaccrc32checker_last_word_error;
					liteethmaccrc32checker_last_error_next_value_ce1 <= 1'd1;
				end
			end
			liteethmaccrc32checker_source_source_payload_error <= (liteethmaccrc32checker_sink_sink_last ? {1{liteethmaccrc32checker_last_word_error}} : liteethmaccrc32checker_syncfifo_source_payload_error);
			liteethmaccrc32checker_error <= ((liteethmaccrc32checker_sink_sink_valid & liteethmaccrc32checker_sink_sink_last) & liteethmaccrc32checker_crc_error);
			if ((liteethmaccrc32checker_sink_sink_valid & liteethmaccrc32checker_sink_sink_ready)) begin
				liteethmaccrc32checker_crc_ce <= 1'd1;
				if ((liteethmaccrc32checker_sink_sink_last & (liteethmaccrc32checker_sink_sink_payload_be > 4'd15))) begin
					rxdatapath_bufferizeendpoints_next_state <= 1'd1;
				end else begin
					if (liteethmaccrc32checker_sink_sink_last) begin
						rxdatapath_bufferizeendpoints_next_state <= 1'd0;
					end
				end
			end
		end
		default: begin
			liteethmaccrc32checker_is_ongoing0 <= 1'd1;
			liteethmaccrc32checker_crc_reset <= 1'd1;
			liteethmaccrc32checker_fifo_reset <= 1'd1;
			rxdatapath_bufferizeendpoints_next_state <= 2'd2;
			liteethmaccrc32checker_is_ongoing2 <= 1'd1;
		end
	endcase
// synthesis translate_off
	dummy_d_20 <= dummy_s;
// synthesis translate_on
end
assign bufferizeendpoints_pipe_valid_sink_ready = ((~bufferizeendpoints_pipe_valid_source_valid) | bufferizeendpoints_pipe_valid_source_ready);
assign bufferizeendpoints_pipe_valid_sink_valid = bufferizeendpoints_sink_sink_valid;
assign bufferizeendpoints_sink_sink_ready = bufferizeendpoints_pipe_valid_sink_ready;
assign bufferizeendpoints_pipe_valid_sink_first = bufferizeendpoints_sink_sink_first;
assign bufferizeendpoints_pipe_valid_sink_last = bufferizeendpoints_sink_sink_last;
assign bufferizeendpoints_pipe_valid_sink_payload_data = bufferizeendpoints_sink_sink_payload_data;
assign bufferizeendpoints_pipe_valid_sink_payload_be = bufferizeendpoints_sink_sink_payload_be;
assign bufferizeendpoints_pipe_valid_sink_payload_error = bufferizeendpoints_sink_sink_payload_error;
assign bufferizeendpoints_source_source_valid = bufferizeendpoints_pipe_valid_source_valid;
assign bufferizeendpoints_pipe_valid_source_ready = bufferizeendpoints_source_source_ready;
assign bufferizeendpoints_source_source_first = bufferizeendpoints_pipe_valid_source_first;
assign bufferizeendpoints_source_source_last = bufferizeendpoints_pipe_valid_source_last;
assign bufferizeendpoints_source_source_payload_data = bufferizeendpoints_pipe_valid_source_payload_data;
assign bufferizeendpoints_source_source_payload_be = bufferizeendpoints_pipe_valid_source_payload_be;
assign bufferizeendpoints_source_source_payload_error = bufferizeendpoints_pipe_valid_source_payload_error;
assign pulsesynchronizer1_o = (pulsesynchronizer1_toggle_o ^ pulsesynchronizer1_toggle_o_r);
assign rx_padding_length_inc = (1'd0 + rx_padding_sink_payload_be);
assign rx_padding_source_valid = rx_padding_sink_valid;
assign rx_padding_sink_ready = rx_padding_source_ready;
assign rx_padding_source_first = rx_padding_sink_first;
assign rx_padding_source_last = rx_padding_sink_last;
assign rx_padding_source_payload_data = rx_padding_sink_payload_data;
assign rx_padding_source_payload_be = rx_padding_sink_payload_be;

// synthesis translate_off
reg dummy_d_21;
// synthesis translate_on
always @(*) begin
	rx_padding_source_payload_error <= 1'd0;
	if (((rx_padding_sink_valid & rx_padding_sink_last) & ((rx_padding_length + rx_padding_length_inc) < 6'd60))) begin
		rx_padding_source_payload_error <= {1{1'd1}};
	end else begin
		rx_padding_source_payload_error <= rx_padding_sink_payload_error;
	end
// synthesis translate_off
	dummy_d_21 <= dummy_s;
// synthesis translate_on
end
assign rx_packet_fifo_sink_sink_valid = rmiirxshim_source_valid;
assign rmiirxshim_source_ready = rx_packet_fifo_sink_sink_ready;
assign rx_packet_fifo_sink_sink_first = rmiirxshim_source_first;
assign rx_packet_fifo_sink_sink_last = rmiirxshim_source_last;
assign rx_packet_fifo_sink_sink_payload_data = rmiirxshim_source_payload_data;
assign rx_packet_fifo_sink_sink_payload_be = rmiirxshim_source_payload_be;
assign rx_packet_fifo_sink_sink_payload_error = rmiirxshim_source_payload_error;
assign rx_cdc_sink_sink_valid = rx_packet_fifo_source_source_valid;
assign rx_packet_fifo_source_source_ready = rx_cdc_sink_sink_ready;
assign rx_cdc_sink_sink_first = rx_packet_fifo_source_source_first;
assign rx_cdc_sink_sink_last = rx_packet_fifo_source_source_last;
assign rx_cdc_sink_sink_payload_data = rx_packet_fifo_source_source_payload_data;
assign rx_cdc_sink_sink_payload_be = rx_packet_fifo_source_source_payload_be;
assign rx_cdc_sink_sink_payload_error = rx_packet_fifo_source_source_payload_error;
assign rx_preamble_sink_valid = rx_cdc_source_source_valid;
assign rx_cdc_source_source_ready = rx_preamble_sink_ready;
assign rx_preamble_sink_first = rx_cdc_source_source_first;
assign rx_preamble_sink_last = rx_cdc_source_source_last;
assign rx_preamble_sink_payload_data = rx_cdc_source_source_payload_data;
assign rx_preamble_sink_payload_be = rx_cdc_source_source_payload_be;
assign rx_preamble_sink_payload_error = rx_cdc_source_source_payload_error;
assign bufferizeendpoints_sink_sink_valid = rx_preamble_source_valid;
assign rx_preamble_source_ready = bufferizeendpoints_sink_sink_ready;
assign bufferizeendpoints_sink_sink_first = rx_preamble_source_first;
assign bufferizeendpoints_sink_sink_last = rx_preamble_source_last;
assign bufferizeendpoints_sink_sink_payload_data = rx_preamble_source_payload_data;
assign bufferizeendpoints_sink_sink_payload_be = rx_preamble_source_payload_be;
assign bufferizeendpoints_sink_sink_payload_error = rx_preamble_source_payload_error;
assign rx_padding_sink_valid = liteethmaccrc32checker_source_source_valid;
assign liteethmaccrc32checker_source_source_ready = rx_padding_sink_ready;
assign rx_padding_sink_first = liteethmaccrc32checker_source_source_first;
assign rx_padding_sink_last = liteethmaccrc32checker_source_source_last;
assign rx_padding_sink_payload_data = liteethmaccrc32checker_source_source_payload_data;
assign rx_padding_sink_payload_be = liteethmaccrc32checker_source_source_payload_be;
assign rx_padding_sink_payload_error = liteethmaccrc32checker_source_source_payload_error;
assign source_valid = rx_padding_source_valid;
assign rx_padding_source_ready = source_ready;
assign source_first = rx_padding_source_first;
assign source_last = rx_padding_source_last;
assign source_payload_data = rx_padding_source_payload_data;
assign source_payload_be = rx_padding_source_payload_be;
assign source_payload_error = rx_padding_source_payload_error;
assign slice_proxy = {tx_crc_value, tx_crc_sink_payload_data[7:0]};
assign txdatapath_write_ext_rst = multiregimpl01;
assign txdatapath_ext_rst_to = multiregimpl11;
assign txdatapath_cdc_produce_rdomain = multiregimpl21;
assign txdatapath_cdc_consume_wdomain = multiregimpl31;
assign toggle_o = multiregimpl41;
assign rx_cdc_write_ext_rst = multiregimpl51;
assign rx_cdc_ext_rst_to = multiregimpl61;
assign rx_cdc_cdc_produce_rdomain = multiregimpl71;
assign rx_cdc_cdc_consume_wdomain = multiregimpl81;
assign pulsesynchronizer0_toggle_o = multiregimpl91;
assign pulsesynchronizer1_toggle_o = multiregimpl101;

always @(posedge eth_rx_clk) begin
	rmiirxshim_crs_dv_d <= rmiirxshim_crs_dv;
	if (rmiirxshim_crs_first) begin
		rmiirxshim_crs_run <= 1'd1;
	end
	if (rmiirxshim_crs_last) begin
		rmiirxshim_crs_run <= 1'd0;
	end
	rmiirxshim_d0 <= rmiirxshim_rx_data;
	rmiirxshim_d1 <= rmiirxshim_d0;
	rmiirxshim_e0 <= rmiirxshim_cap;
	rmiirxshim_e1 <= rmiirxshim_e0;
	if ((rmiirxshim_e1 & (rmiirxshim_mode == 1'd0))) begin
		rmiirxshim_sreg <= {rmiirxshim_sreg[5:0], rmiirxshim_d1};
	end
	if ((rmiirxshim_mode == 1'd0)) begin
		if (rmiirxshim_sfd_hit) begin
			rmiirxshim_mode <= 1'd1;
			rmiirxshim_pre_cnt <= 1'd0;
			rmiirxshim_bits <= 1'd0;
		end
	end else begin
		if ((rmiirxshim_mode == 1'd1)) begin
			if (rmiirxshim_consume) begin
				if ((rmiirxshim_pre_cnt == 3'd7)) begin
					rmiirxshim_mode <= 2'd2;
				end else begin
					rmiirxshim_pre_cnt <= (rmiirxshim_pre_cnt + 1'd1);
				end
			end
		end
	end
	if (((rmiirxshim_mode == 2'd2) & rmiirxshim_consume)) begin
		if ((rmiirxshim_pushing & (~rmiirxshim_f1_valid))) begin
			rmiirxshim_f0_valid <= 1'd1;
			rmiirxshim_f0_data <= rmiirxshim_new_byte;
			rmiirxshim_f0_last <= rmiirxshim_crs_last;
			rmiirxshim_f1_valid <= 1'd0;
		end else begin
			if (rmiirxshim_pushing) begin
				rmiirxshim_f0_valid <= 1'd1;
				rmiirxshim_f0_data <= rmiirxshim_f1_data;
				rmiirxshim_f0_last <= rmiirxshim_f1_last;
				rmiirxshim_f1_valid <= 1'd1;
				rmiirxshim_f1_data <= rmiirxshim_new_byte;
				rmiirxshim_f1_last <= rmiirxshim_crs_last;
			end else begin
				rmiirxshim_f0_valid <= rmiirxshim_f1_valid;
				rmiirxshim_f0_data <= rmiirxshim_f1_data;
				rmiirxshim_f0_last <= rmiirxshim_f1_last;
				rmiirxshim_f1_valid <= 1'd0;
			end
		end
	end else begin
		if ((rmiirxshim_pushing & (~(((rmiirxshim_mode == 2'd2) & rmiirxshim_f0_valid) & rmiirxshim_f1_valid)))) begin
			if ((~rmiirxshim_f0_valid)) begin
				rmiirxshim_f0_valid <= 1'd1;
				rmiirxshim_f0_data <= rmiirxshim_new_byte;
				rmiirxshim_f0_last <= rmiirxshim_crs_last;
			end else begin
				if ((~rmiirxshim_f1_valid)) begin
					rmiirxshim_f1_valid <= 1'd1;
					rmiirxshim_f1_data <= rmiirxshim_new_byte;
					rmiirxshim_f1_last <= rmiirxshim_crs_last;
				end
			end
		end
	end
	if (((rmiirxshim_mode != 1'd0) & rmiirxshim_e1)) begin
		rmiirxshim_shift <= rmiirxshim_new_byte;
		if ((rmiirxshim_bits == 2'd3)) begin
			rmiirxshim_bits <= 1'd0;
		end else begin
			rmiirxshim_bits <= (rmiirxshim_bits + 1'd1);
		end
	end else begin
		if ((~rmiirxshim_e1)) begin
			rmiirxshim_bits <= 1'd0;
		end
	end
	if ((rmiirxshim_consume & rmiirxshim_source_last)) begin
		rmiirxshim_mode <= 1'd0;
		rmiirxshim_sreg <= 1'd0;
		rmiirxshim_bits <= 1'd0;
		rmiirxshim_f0_valid <= 1'd0;
		rmiirxshim_f1_valid <= 1'd0;
	end
	if (rx_packet_fifo_sink_sink_valid) begin
		rx_packet_fifo_invalid <= (rx_packet_fifo_invalid | rx_packet_fifo_discard);
		if (((~rx_packet_fifo_overflow) & (~rx_packet_fifo_full))) begin
			rx_packet_fifo_wr_ptr <= (rx_packet_fifo_wr_ptr + 1'd1);
		end else begin
			rx_packet_fifo_overflow <= 1'd1;
		end
		if (rx_packet_fifo_sink_sink_last) begin
			rx_packet_fifo_overflow <= 1'd0;
			rx_packet_fifo_invalid <= 1'd0;
			if (rx_packet_fifo_commit) begin
				rx_packet_fifo_commit_ptr <= (rx_packet_fifo_wr_ptr + 1'd1);
			end else begin
				rx_packet_fifo_wr_ptr <= rx_packet_fifo_commit_ptr;
			end
		end
	end
	if (rx_packet_fifo_fetch) begin
		rx_packet_fifo_rd_ptr <= (rx_packet_fifo_rd_ptr + 1'd1);
		rx_packet_fifo_out_valid <= 1'd1;
	end else begin
		if ((rx_packet_fifo_source_source_valid & rx_packet_fifo_source_source_ready)) begin
			rx_packet_fifo_out_valid <= 1'd0;
		end
	end
	if (((rx_packet_fifo_param_fifo_syncfifo_we & rx_packet_fifo_param_fifo_syncfifo_writable) & (~rx_packet_fifo_param_fifo_replace))) begin
		rx_packet_fifo_param_fifo_produce <= (rx_packet_fifo_param_fifo_produce + 1'd1);
	end
	if (rx_packet_fifo_param_fifo_do_read) begin
		rx_packet_fifo_param_fifo_consume <= (rx_packet_fifo_param_fifo_consume + 1'd1);
	end
	if (((rx_packet_fifo_param_fifo_syncfifo_we & rx_packet_fifo_param_fifo_syncfifo_writable) & (~rx_packet_fifo_param_fifo_replace))) begin
		if ((~rx_packet_fifo_param_fifo_do_read)) begin
			rx_packet_fifo_param_fifo_level <= (rx_packet_fifo_param_fifo_level + 1'd1);
		end
	end else begin
		if (rx_packet_fifo_param_fifo_do_read) begin
			rx_packet_fifo_param_fifo_level <= (rx_packet_fifo_param_fifo_level - 1'd1);
		end
	end
	if (i) begin
		toggle_i <= (~toggle_i);
	end
	rx_cdc_cdc_graycounter0_q_binary <= rx_cdc_cdc_graycounter0_q_next_binary;
	rx_cdc_cdc_graycounter0_q <= rx_cdc_cdc_graycounter0_q_next;
	if (rx_cdc_cdc_reset_write) begin
		rx_cdc_cdc_graycounter0_q <= 6'd0;
		rx_cdc_cdc_graycounter0_q_binary <= 6'd0;
	end
	if (eth_rx_rst) begin
		rmiirxshim_crs_run <= 1'd0;
		rmiirxshim_crs_dv_d <= 1'd0;
		rmiirxshim_d0 <= 2'd0;
		rmiirxshim_d1 <= 2'd0;
		rmiirxshim_e0 <= 1'd0;
		rmiirxshim_e1 <= 1'd0;
		rmiirxshim_mode <= 2'd0;
		rmiirxshim_f0_valid <= 1'd0;
		rmiirxshim_f0_data <= 8'd0;
		rmiirxshim_f0_last <= 1'd0;
		rmiirxshim_f1_valid <= 1'd0;
		rmiirxshim_f1_data <= 8'd0;
		rmiirxshim_f1_last <= 1'd0;
		rmiirxshim_sreg <= 8'd0;
		rmiirxshim_pre_cnt <= 3'd0;
		rmiirxshim_shift <= 8'd0;
		rmiirxshim_bits <= 2'd0;
		rx_packet_fifo_param_fifo_level <= 3'd0;
		rx_packet_fifo_param_fifo_produce <= 2'd0;
		rx_packet_fifo_param_fifo_consume <= 2'd0;
		rx_packet_fifo_wr_ptr <= 12'd0;
		rx_packet_fifo_commit_ptr <= 12'd0;
		rx_packet_fifo_rd_ptr <= 12'd0;
		rx_packet_fifo_out_valid <= 1'd0;
		rx_packet_fifo_overflow <= 1'd0;
		rx_packet_fifo_invalid <= 1'd0;
		rx_cdc_cdc_graycounter0_q <= 6'd0;
		rx_cdc_cdc_graycounter0_q_binary <= 6'd0;
	end
	multiregimpl50 <= sys_rst;
	multiregimpl51 <= multiregimpl50;
	multiregimpl80 <= rx_cdc_cdc_graycounter1_q;
	multiregimpl81 <= multiregimpl80;
end

always @(posedge eth_tx_clk) begin
	if ((rmiitxshim_sink_valid & rmiitxshim_sink_ready)) begin
		rmiitxshim_shift <= rmiitxshim_sink_payload_data;
		rmiitxshim_count <= 3'd4;
	end else begin
		if ((rmiitxshim_count != 1'd0)) begin
			rmiitxshim_shift <= (rmiitxshim_shift >>> 2'd2);
			rmiitxshim_count <= (rmiitxshim_count - 1'd1);
		end
	end
	txdatapath_cdc_graycounter1_q_binary <= txdatapath_cdc_graycounter1_q_next_binary;
	txdatapath_cdc_graycounter1_q <= txdatapath_cdc_graycounter1_q_next;
	if (txdatapath_cdc_reset_read) begin
		txdatapath_cdc_graycounter1_q <= 6'd0;
		txdatapath_cdc_graycounter1_q_binary <= 6'd0;
	end
	if (tx_packet_fifo_payload_fifo_syncfifo_re) begin
		tx_packet_fifo_payload_fifo_readable <= 1'd1;
	end else begin
		if (tx_packet_fifo_payload_fifo_re) begin
			tx_packet_fifo_payload_fifo_readable <= 1'd0;
		end
	end
	if (((tx_packet_fifo_payload_fifo_syncfifo_we & tx_packet_fifo_payload_fifo_syncfifo_writable) & (~tx_packet_fifo_payload_fifo_replace))) begin
		tx_packet_fifo_payload_fifo_produce <= (tx_packet_fifo_payload_fifo_produce + 1'd1);
	end
	if (tx_packet_fifo_payload_fifo_do_read) begin
		tx_packet_fifo_payload_fifo_consume <= (tx_packet_fifo_payload_fifo_consume + 1'd1);
	end
	if (((tx_packet_fifo_payload_fifo_syncfifo_we & tx_packet_fifo_payload_fifo_syncfifo_writable) & (~tx_packet_fifo_payload_fifo_replace))) begin
		if ((~tx_packet_fifo_payload_fifo_do_read)) begin
			tx_packet_fifo_payload_fifo_level0 <= (tx_packet_fifo_payload_fifo_level0 + 1'd1);
		end
	end else begin
		if (tx_packet_fifo_payload_fifo_do_read) begin
			tx_packet_fifo_payload_fifo_level0 <= (tx_packet_fifo_payload_fifo_level0 - 1'd1);
		end
	end
	if (tx_packet_fifo_param_fifo_syncfifo_re) begin
		tx_packet_fifo_param_fifo_readable <= 1'd1;
	end else begin
		if (tx_packet_fifo_param_fifo_re) begin
			tx_packet_fifo_param_fifo_readable <= 1'd0;
		end
	end
	if (((tx_packet_fifo_param_fifo_syncfifo_we & tx_packet_fifo_param_fifo_syncfifo_writable) & (~tx_packet_fifo_param_fifo_replace))) begin
		if ((tx_packet_fifo_param_fifo_produce == 3'd4)) begin
			tx_packet_fifo_param_fifo_produce <= 1'd0;
		end else begin
			tx_packet_fifo_param_fifo_produce <= (tx_packet_fifo_param_fifo_produce + 1'd1);
		end
	end
	if (tx_packet_fifo_param_fifo_do_read) begin
		if ((tx_packet_fifo_param_fifo_consume == 3'd4)) begin
			tx_packet_fifo_param_fifo_consume <= 1'd0;
		end else begin
			tx_packet_fifo_param_fifo_consume <= (tx_packet_fifo_param_fifo_consume + 1'd1);
		end
	end
	if (((tx_packet_fifo_param_fifo_syncfifo_we & tx_packet_fifo_param_fifo_syncfifo_writable) & (~tx_packet_fifo_param_fifo_replace))) begin
		if ((~tx_packet_fifo_param_fifo_do_read)) begin
			tx_packet_fifo_param_fifo_level0 <= (tx_packet_fifo_param_fifo_level0 + 1'd1);
		end
	end else begin
		if (tx_packet_fifo_param_fifo_do_read) begin
			tx_packet_fifo_param_fifo_level0 <= (tx_packet_fifo_param_fifo_level0 - 1'd1);
		end
	end
	if (tx_packet_fifo_reset) begin
		tx_packet_fifo_payload_fifo_readable <= 1'd0;
		tx_packet_fifo_payload_fifo_level0 <= 12'd0;
		tx_packet_fifo_payload_fifo_produce <= 11'd0;
		tx_packet_fifo_payload_fifo_consume <= 11'd0;
		tx_packet_fifo_param_fifo_readable <= 1'd0;
		tx_packet_fifo_param_fifo_level0 <= 3'd0;
		tx_packet_fifo_param_fifo_produce <= 3'd0;
		tx_packet_fifo_param_fifo_consume <= 3'd0;
	end
	txdatapath_resetinserter_state <= txdatapath_resetinserter_next_state;
	if (tx_gap_counter_clockdomainsrenamer3_next_value_ce) begin
		tx_gap_counter <= tx_gap_counter_clockdomainsrenamer3_next_value;
	end
	if (tx_gap_reset) begin
		txdatapath_resetinserter_state <= 1'd0;
	end
	if (eth_tx_rst) begin
		rmiitxshim_shift <= 8'd0;
		rmiitxshim_count <= 3'd0;
		txdatapath_cdc_graycounter1_q <= 6'd0;
		txdatapath_cdc_graycounter1_q_binary <= 6'd0;
		tx_packet_fifo_payload_fifo_readable <= 1'd0;
		tx_packet_fifo_payload_fifo_level0 <= 12'd0;
		tx_packet_fifo_payload_fifo_produce <= 11'd0;
		tx_packet_fifo_payload_fifo_consume <= 11'd0;
		tx_packet_fifo_param_fifo_readable <= 1'd0;
		tx_packet_fifo_param_fifo_level0 <= 3'd0;
		tx_packet_fifo_param_fifo_produce <= 3'd0;
		tx_packet_fifo_param_fifo_consume <= 3'd0;
		txdatapath_resetinserter_state <= 1'd0;
	end
	multiregimpl10 <= txdatapath_write_rst;
	multiregimpl11 <= multiregimpl10;
	multiregimpl20 <= txdatapath_cdc_graycounter0_q;
	multiregimpl21 <= multiregimpl20;
end

always @(posedge sys_clk) begin
	txdatapath_liteethmacpaddinginserter_state <= txdatapath_liteethmacpaddinginserter_next_state;
	if (tx_padding_counter_clockdomainsrenamer0_next_value_ce) begin
		tx_padding_counter <= tx_padding_counter_clockdomainsrenamer0_next_value;
	end
	if (tx_crc_is_ongoing0) begin
		tx_crc_cnt <= 2'd3;
	end else begin
		if ((tx_crc_is_ongoing1 & (~tx_crc_cnt_done))) begin
			tx_crc_cnt <= (tx_crc_cnt - tx_crc_source_ready);
		end
	end
	if (tx_crc_ce) begin
		tx_crc_reg <= tx_crc_crc_next;
	end
	if (tx_crc_reset) begin
		tx_crc_reg <= 32'd4294967295;
	end
	txdatapath_bufferizeendpoints_state <= txdatapath_bufferizeendpoints_next_state;
	if (tx_crc_crc_packet_clockdomainsrenamer1_next_value_ce0) begin
		tx_crc_crc_packet <= tx_crc_crc_packet_clockdomainsrenamer1_next_value0;
	end
	if (tx_crc_be0_clockdomainsrenamer1_next_value_ce1) begin
		tx_crc_be0 <= tx_crc_be0_clockdomainsrenamer1_next_value1;
	end
	if (((~tx_crc_pipe_valid_source_valid) | tx_crc_pipe_valid_source_ready)) begin
		tx_crc_pipe_valid_source_valid <= tx_crc_pipe_valid_sink_valid;
		tx_crc_pipe_valid_source_first <= tx_crc_pipe_valid_sink_first;
		tx_crc_pipe_valid_source_last <= tx_crc_pipe_valid_sink_last;
		tx_crc_pipe_valid_source_payload_data <= tx_crc_pipe_valid_sink_payload_data;
		tx_crc_pipe_valid_source_payload_be <= tx_crc_pipe_valid_sink_payload_be;
		tx_crc_pipe_valid_source_payload_error <= tx_crc_pipe_valid_sink_payload_error;
	end
	txdatapath_liteethmacpreambleinserter_state <= txdatapath_liteethmacpreambleinserter_next_state;
	if (tx_preamble_count_clockdomainsrenamer2_next_value_ce) begin
		tx_preamble_count <= tx_preamble_count_clockdomainsrenamer2_next_value;
	end
	txdatapath_cdc_graycounter0_q_binary <= txdatapath_cdc_graycounter0_q_next_binary;
	txdatapath_cdc_graycounter0_q <= txdatapath_cdc_graycounter0_q_next;
	if (txdatapath_cdc_reset_write) begin
		txdatapath_cdc_graycounter0_q <= 6'd0;
		txdatapath_cdc_graycounter0_q_binary <= 6'd0;
	end
	if (o) begin
		drops_status <= (drops_status + 1'd1);
	end
	if (pulsesynchronizer0_o) begin
		preamble_errors_status <= (preamble_errors_status + 1'd1);
	end
	if (pulsesynchronizer1_o) begin
		crc_errors_status <= (crc_errors_status + 1'd1);
	end
	toggle_o_r <= toggle_o;
	rx_cdc_cdc_graycounter1_q_binary <= rx_cdc_cdc_graycounter1_q_next_binary;
	rx_cdc_cdc_graycounter1_q <= rx_cdc_cdc_graycounter1_q_next;
	if (rx_cdc_cdc_reset_read) begin
		rx_cdc_cdc_graycounter1_q <= 6'd0;
		rx_cdc_cdc_graycounter1_q_binary <= 6'd0;
	end
	rxdatapath_liteethmacpreamblechecker_state <= rxdatapath_liteethmacpreamblechecker_next_state;
	if (pulsesynchronizer0_i) begin
		pulsesynchronizer0_toggle_i <= (~pulsesynchronizer0_toggle_i);
	end
	pulsesynchronizer0_toggle_o_r <= pulsesynchronizer0_toggle_o;
	if (liteethmaccrc32checker_is_ongoing2) begin
		liteethmaccrc32checker_packet_error <= 1'd0;
	end else begin
		if ((liteethmaccrc32checker_sink_sink_valid & liteethmaccrc32checker_sink_sink_ready)) begin
			liteethmaccrc32checker_packet_error <= (liteethmaccrc32checker_packet_error | ((liteethmaccrc32checker_sink_sink_payload_error & liteethmaccrc32checker_sink_sink_payload_be) != 1'd0));
		end
	end
	if (liteethmaccrc32checker_crc_ce) begin
		liteethmaccrc32checker_crc_reg <= liteethmaccrc32checker_crc_crc_next;
	end
	if (liteethmaccrc32checker_crc_reset) begin
		liteethmaccrc32checker_crc_reg <= 32'd4294967295;
	end
	if (((liteethmaccrc32checker_syncfifo_syncfifo_we & liteethmaccrc32checker_syncfifo_syncfifo_writable) & (~liteethmaccrc32checker_syncfifo_replace))) begin
		if ((liteethmaccrc32checker_syncfifo_produce == 3'd4)) begin
			liteethmaccrc32checker_syncfifo_produce <= 1'd0;
		end else begin
			liteethmaccrc32checker_syncfifo_produce <= (liteethmaccrc32checker_syncfifo_produce + 1'd1);
		end
	end
	if (liteethmaccrc32checker_syncfifo_do_read) begin
		if ((liteethmaccrc32checker_syncfifo_consume == 3'd4)) begin
			liteethmaccrc32checker_syncfifo_consume <= 1'd0;
		end else begin
			liteethmaccrc32checker_syncfifo_consume <= (liteethmaccrc32checker_syncfifo_consume + 1'd1);
		end
	end
	if (((liteethmaccrc32checker_syncfifo_syncfifo_we & liteethmaccrc32checker_syncfifo_syncfifo_writable) & (~liteethmaccrc32checker_syncfifo_replace))) begin
		if ((~liteethmaccrc32checker_syncfifo_do_read)) begin
			liteethmaccrc32checker_syncfifo_level <= (liteethmaccrc32checker_syncfifo_level + 1'd1);
		end
	end else begin
		if (liteethmaccrc32checker_syncfifo_do_read) begin
			liteethmaccrc32checker_syncfifo_level <= (liteethmaccrc32checker_syncfifo_level - 1'd1);
		end
	end
	if (liteethmaccrc32checker_fifo_reset) begin
		liteethmaccrc32checker_syncfifo_level <= 3'd0;
		liteethmaccrc32checker_syncfifo_produce <= 3'd0;
		liteethmaccrc32checker_syncfifo_consume <= 3'd0;
	end
	rxdatapath_bufferizeendpoints_state <= rxdatapath_bufferizeendpoints_next_state;
	if (liteethmaccrc32checker_be_next_value_ce0) begin
		liteethmaccrc32checker_be <= liteethmaccrc32checker_be_next_value0;
	end
	if (liteethmaccrc32checker_last_error_next_value_ce1) begin
		liteethmaccrc32checker_last_error <= liteethmaccrc32checker_last_error_next_value1;
	end
	if (((~bufferizeendpoints_pipe_valid_source_valid) | bufferizeendpoints_pipe_valid_source_ready)) begin
		bufferizeendpoints_pipe_valid_source_valid <= bufferizeendpoints_pipe_valid_sink_valid;
		bufferizeendpoints_pipe_valid_source_first <= bufferizeendpoints_pipe_valid_sink_first;
		bufferizeendpoints_pipe_valid_source_last <= bufferizeendpoints_pipe_valid_sink_last;
		bufferizeendpoints_pipe_valid_source_payload_data <= bufferizeendpoints_pipe_valid_sink_payload_data;
		bufferizeendpoints_pipe_valid_source_payload_be <= bufferizeendpoints_pipe_valid_sink_payload_be;
		bufferizeendpoints_pipe_valid_source_payload_error <= bufferizeendpoints_pipe_valid_sink_payload_error;
	end
	if (pulsesynchronizer1_i) begin
		pulsesynchronizer1_toggle_i <= (~pulsesynchronizer1_toggle_i);
	end
	pulsesynchronizer1_toggle_o_r <= pulsesynchronizer1_toggle_o;
	if ((rx_padding_sink_valid & rx_padding_sink_ready)) begin
		if (rx_padding_sink_last) begin
			rx_padding_length <= 1'd0;
		end else begin
			rx_padding_length <= (rx_padding_length + rx_padding_length_inc);
		end
	end
	if (sys_rst) begin
		tx_padding_counter <= 16'd0;
		tx_crc_reg <= 32'd4294967295;
		tx_crc_cnt <= 2'd3;
		tx_crc_pipe_valid_source_valid <= 1'd0;
		txdatapath_cdc_graycounter0_q <= 6'd0;
		txdatapath_cdc_graycounter0_q_binary <= 6'd0;
		preamble_errors_status <= 32'd0;
		crc_errors_status <= 32'd0;
		drops_status <= 32'd0;
		rx_cdc_cdc_graycounter1_q <= 6'd0;
		rx_cdc_cdc_graycounter1_q_binary <= 6'd0;
		liteethmaccrc32checker_crc_reg <= 32'd4294967295;
		liteethmaccrc32checker_syncfifo_level <= 3'd0;
		liteethmaccrc32checker_syncfifo_produce <= 3'd0;
		liteethmaccrc32checker_syncfifo_consume <= 3'd0;
		liteethmaccrc32checker_last_error <= 1'd0;
		liteethmaccrc32checker_packet_error <= 1'd0;
		bufferizeendpoints_pipe_valid_source_valid <= 1'd0;
		rx_padding_length <= 11'd0;
		txdatapath_liteethmacpaddinginserter_state <= 1'd0;
		txdatapath_bufferizeendpoints_state <= 2'd0;
		txdatapath_liteethmacpreambleinserter_state <= 2'd0;
		rxdatapath_liteethmacpreamblechecker_state <= 1'd0;
		rxdatapath_bufferizeendpoints_state <= 2'd0;
	end
	multiregimpl00 <= eth_tx_rst;
	multiregimpl01 <= multiregimpl00;
	multiregimpl30 <= txdatapath_cdc_graycounter1_q;
	multiregimpl31 <= multiregimpl30;
	multiregimpl40 <= toggle_i;
	multiregimpl41 <= multiregimpl40;
	multiregimpl60 <= rx_cdc_write_rst;
	multiregimpl61 <= multiregimpl60;
	multiregimpl70 <= rx_cdc_cdc_graycounter0_q;
	multiregimpl71 <= multiregimpl70;
	multiregimpl90 <= pulsesynchronizer0_toggle_i;
	multiregimpl91 <= multiregimpl90;
	multiregimpl100 <= pulsesynchronizer1_toggle_i;
	multiregimpl101 <= multiregimpl100;
end

reg [11:0] storage[0:31];
reg [4:0] memadr;
always @(posedge sys_clk) begin : mem_write_block
	if (txdatapath_cdc_wrport_we)
		storage[txdatapath_cdc_wrport_adr] <= txdatapath_cdc_wrport_dat_w;
end

always @(posedge eth_tx_clk) begin : mem_write_block_1
	memadr <= txdatapath_cdc_rdport_adr;
end

assign txdatapath_cdc_rdport_dat_r = storage[memadr];

reg [11:0] storage_1[0:2047];
reg [11:0] memdat;
always @(posedge eth_tx_clk) begin : mem_write_block_2
	if (tx_packet_fifo_payload_fifo_wrport_we)
		storage_1[tx_packet_fifo_payload_fifo_wrport_adr] <= tx_packet_fifo_payload_fifo_wrport_dat_w;
end

always @(posedge eth_tx_clk) begin : mem_write_block_3
	if (tx_packet_fifo_payload_fifo_rdport_re)
		memdat <= storage_1[tx_packet_fifo_payload_fifo_rdport_adr];
end

assign tx_packet_fifo_payload_fifo_rdport_dat_r = memdat;

reg [2:0] storage_2[0:4];
reg [2:0] memdat_1;
always @(posedge eth_tx_clk) begin : mem_write_block_4
	if (tx_packet_fifo_param_fifo_wrport_we)
		storage_2[tx_packet_fifo_param_fifo_wrport_adr] <= tx_packet_fifo_param_fifo_wrport_dat_w;
end

always @(posedge eth_tx_clk) begin : mem_write_block_5
	if (tx_packet_fifo_param_fifo_rdport_re)
		memdat_1 <= storage_2[tx_packet_fifo_param_fifo_rdport_adr];
end

assign tx_packet_fifo_param_fifo_rdport_dat_r = memdat_1;

reg [2:0] storage_3[0:3];
always @(posedge eth_rx_clk) begin : mem_write_block_6
	if (rx_packet_fifo_param_fifo_wrport_we)
		storage_3[rx_packet_fifo_param_fifo_wrport_adr] <= rx_packet_fifo_param_fifo_wrport_dat_w;
end

always @(posedge eth_rx_clk) begin : mem_write_block_7
end

assign rx_packet_fifo_param_fifo_rdport_dat_r = storage_3[rx_packet_fifo_param_fifo_rdport_adr];

reg [11:0] mem[0:2047];
reg [10:0] memadr_1;
reg [10:0] memadr_2;
always @(posedge eth_rx_clk) begin : mem_write_block_8
	if (rx_packet_fifo_wr_port_we)
		mem[rx_packet_fifo_wr_port_adr] <= rx_packet_fifo_wr_port_dat_w;
	memadr_1 <= rx_packet_fifo_wr_port_adr;
end

always @(posedge eth_rx_clk) begin : mem_write_block_9
	if (rx_packet_fifo_rd_port_re)
		memadr_2 <= rx_packet_fifo_rd_port_adr;
end

assign rx_packet_fifo_wr_port_dat_r = mem[memadr_1];
assign rx_packet_fifo_rd_port_dat_r = mem[memadr_2];

reg [11:0] storage_4[0:31];
reg [4:0] memadr_3;
always @(posedge eth_rx_clk) begin : mem_write_block_10
	if (rx_cdc_cdc_wrport_we)
		storage_4[rx_cdc_cdc_wrport_adr] <= rx_cdc_cdc_wrport_dat_w;
end

always @(posedge sys_clk) begin : mem_write_block_11
	memadr_3 <= rx_cdc_cdc_rdport_adr;
end

assign rx_cdc_cdc_rdport_dat_r = storage_4[memadr_3];

reg [11:0] storage_5[0:4];
always @(posedge sys_clk) begin : mem_write_block_12
	if (liteethmaccrc32checker_syncfifo_wrport_we)
		storage_5[liteethmaccrc32checker_syncfifo_wrport_adr] <= liteethmaccrc32checker_syncfifo_wrport_dat_w;
end

always @(posedge sys_clk) begin : mem_write_block_13
end

assign liteethmaccrc32checker_syncfifo_rdport_dat_r = storage_5[liteethmaccrc32checker_syncfifo_rdport_adr];

endmodule
